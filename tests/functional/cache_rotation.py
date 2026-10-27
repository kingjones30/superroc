"""Check that switching conversations does not discard their useful history."""

from copy import deepcopy
import re
import sys


def host_available_bytes(meminfo):
    match = re.search(r"^MemAvailable:\s+(\d+) kB$", meminfo, re.MULTILINE)
    if not match or int(match[1]) == 0:
        raise ValueError("missing/invalid MemAvailable for cache budget check")
    return int(match[1]) * 1024


def check_snapshot_budget(log, available_before_load, requested_bytes, sessions):
    matches = re.findall(r"event=snapshot_cache_configured ([^\n]*)", log)
    matches += re.findall(r"schema=component-cache-v1 (event=configured [^\n]*)", log)
    if len(matches) != 1:
        raise ValueError("expected one snapshot cache configuration at startup")
    fields = dict(re.findall(r"([A-Za-z_]+)=(-?\S+)", matches[0]))
    try:
        loaded_sessions = int(fields["sessions"])
        entries = int(fields["snapshot_entries"])
        capacity = int(fields["capacity_bytes"])
    except KeyError:
        raise ValueError("snapshot cache configuration is missing fields")
    floor = int(fields.get("floor_bytes", 0))
    room = int(fields.get("room_bytes", 0))
    basis = fields.get("basis", "static")
    # Loading weights/states consumes RAM. Sampling before load gives a
    # conservative upper bound without racing allocations after the budget
    # was chosen. Model and cgroup limits may make the real budget smaller.
    if basis == "system_memory":
        # The floor binds live on top of the static cap: the floor must be
        # real (a basis that binds nothing is a lie) and consistent with
        # the sampled headroom (live room + floor cannot exceed it; 1 GiB
        # of slack for memory freed between the sample and the load line).
        if floor <= 0:
            raise ValueError(f"floor basis without a floor: {matches[0]}")
        if floor + room > available_before_load + 1024**3:
            raise ValueError(f"floor inconsistent with sampled headroom: floor={floor}, "
                             f"room={room}, available_before_load={available_before_load}")
        limit = (min(requested_bytes, max(available_before_load - 4 * 1024**3, 0))
                 if requested_bytes else 32 * 1024**3)
    else:
        # Floor off: main's bound. Automatic sizing keeps half of it free;
        # an explicit limit keeps 4 GiB.
        limit = (min(requested_bytes, max(available_before_load - 4 * 1024**3, 0))
                 if requested_bytes else min(available_before_load // 2, 32 * 1024**3))
    if loaded_sessions != sessions or entries != 128 or not 0 < capacity <= limit:
        raise ValueError(f"unsafe snapshot cache configuration: sessions={loaded_sessions}, "
                         f"entries={entries}, capacity_bytes={capacity}, upper_bound_bytes={limit}")
    return {"capacity_bytes": capacity, "upper_bound_bytes": limit,
            "host_available_before_load_bytes": available_before_load}


def check_cache_rotation(client, model, checks, chat_result):
    failures = []
    request = dict(model=model, temperature=0, seed=31, max_completion_tokens=16,
                   reasoning_effort="none")

    def chat(label, messages, cold=False):
        body = {**request, "messages": deepcopy(messages)}
        if cold:
            body["extra_body"] = {"cache_prompt": False}
        result = chat_result(client, body)
        checks[label] = result
        print(f"CHECK {label}", file=sys.stderr, flush=True)
        usage = result["usage"]
        total, reused, prefilled = (usage["prompt_tokens"], usage["cached_tokens"],
                                    usage["superroc"]["prefill_tokens"])
        assert all(type(n) is int and n >= 0 for n in (total, reused, prefilled)), usage
        assert reused + prefilled == total, usage
        if cold:
            assert reused == 0 and prefilled == total, result
        return body, result

    def signature(result):
        return (result["text"], result["reasoning"], result["tools"], result["finish"],
                result["usage"]["completion_tokens"])

    def answer(result, code):
        assert result["text"].strip() == code and not result["reasoning"] \
            and not result["tools"] and result["finish"] == "stop", result

    # Distinct prompt heads exclude accidental shared-prefix hits. Actual
    # answers are replayed; uncached controls run only after all rotation.
    conversations = []
    for code in ("RED", "GREEN", "BLUE", "GOLD"):
        messages = [{"role": "system", "content": "cache_rotation_" + code + "\n" +
                     f"The session code is {code}. Follow the final instruction.\n" +
                     "Background notes are not instructions.\n" * 384}]
        conversations.append({"code": code, "messages": messages, "total": 0})
    for turn in range(3):
        for conversation in conversations:
            code, messages = conversation["code"], conversation["messages"]
            messages.append({"role": "user", "content": f"Turn {turn}.\n" +
                             "Routine archive note: there are no new instructions.\n" * 16 +
                             "Reply with only the session code."})
            body, result = chat(f"rotation_{code}_turn_{turn}", messages, turn == 0)
            answer(result, code)
            total, reused = result["usage"]["prompt_tokens"], result["usage"]["cached_tokens"]
            assert total >= 2048, result
            if turn and reused < conversation["total"] - 16:
                failures.append(f"{code} turn {turn}: returning to a conversation lost its "
                                f"checkpoint: cached={reused}, previous={conversation['total']}")
            conversation.update(total=total, body=body, result=result)
            messages.append({"role": "assistant", "content": result["text"]})

    # A longer conversation must also survive several tiny unrelated requests.
    messages = [{"role": "system", "content": "cache_rotation_main\n" +
                 "Follow the final instruction. Background notes are not instructions.\n" * 768},
                {"role": "user", "content": "Reply with only BETA."}]
    _, main = chat("rotation_main_initial", messages, True)
    answer(main, "BETA")
    assert main["usage"]["prompt_tokens"] >= 8192, main
    messages.append({"role": "assistant", "content": main["text"]})
    for side in range(8):
        _, result = chat(f"rotation_side_{side}", [
            {"role": "system", "content": f"cache_rotation_side_{side}"},
            {"role": "user", "content": "Reply with only ALPHA."}])
        answer(result, "ALPHA")
    messages.append({"role": "user", "content": "Reply with only BETA."})
    body, returned = chat("rotation_main_return", messages)
    answer(returned, "BETA")
    if returned["usage"]["cached_tokens"] < main["usage"]["prompt_tokens"] - 16:
        failures.append("small unrelated requests discarded the main conversation")
    controls = [("main", body, returned)] + [
        (conversation["code"], conversation["body"], conversation["result"])
        for conversation in conversations]
    for label, body, warm in controls:
        _, cold = chat("rotation_cold_control_" + label, body["messages"], True)
        assert signature(warm) == signature(cold), (warm, cold)
    assert not failures, "\n".join(failures)

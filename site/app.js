// Copy buttons for code blocks. No dependencies.
document.querySelectorAll(".code .copy").forEach((btn) => {
  btn.addEventListener("click", async () => {
    const code = btn.parentElement.querySelector("code").innerText;
    try {
      await navigator.clipboard.writeText(code);
      btn.textContent = "Copied";
    } catch (e) {
      btn.textContent = "Select";
    }
    setTimeout(() => (btn.textContent = "Copy"), 1600);
  });
});

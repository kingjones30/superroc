{
  lib,
  newScope,
  releaseVersion ? null,
  revision,
}:

lib.makeScope newScope (self: {
  superroc = self.callPackage ./package.nix { inherit releaseVersion revision; };
  mkServe = self.callPackage ./mk-serve.nix { };
})

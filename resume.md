# Resume

Discussing the SFNO architecture, specifically how channel mixing works in the spectral filter.

The conversation covered:
1. The two channel mixing operations in each FNO block: the MLP (512→1024→512 pointwise) and the spectral filter
2. The spectral filter uses weight tensor `W` of shape `[modes_lat, out_channels, in_channels]` = `[45, 512, 512]`
3. For each spherical harmonic degree ℓ, an independent `[512, 512]` matrix mixes channels — implemented via einsum `"bgixy,gxoi->bgoxy"`
4. The weights are shared across all m (longitudinal wavenumbers), making the filter isotropic in longitude

**Next step**: Read the SFNO paper (`~/Desktop/Arches/SFNO.pdf`) to verify whether this dhconv / ℓ-only weighting matches what the paper describes. Could not access the file due to macOS permissions on the tmux session. Restart tmux and then run:
```
cp ~/Desktop/Arches/SFNO.pdf /tmp/SFNO.pdf
```
Then ask Claude to read `/tmp/SFNO.pdf` pages covering the spectral convolution definition (look for the equation defining the SFNO layer, likely in Section 3).

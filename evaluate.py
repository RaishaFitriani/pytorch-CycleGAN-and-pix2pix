import sys, glob, os, pandas as pd, torch, lpips
from skimage import io, color
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
fake_dir, real_dir, out_csv = sys.argv[1:4]
loss_fn = lpips.LPIPS(net='alex')
to_t = lambda im: torch.from_numpy(im).permute(2, 0, 1)[None].float() / 127.5 - 1
rows = []
for f in sorted(glob.glob(f'{fake_dir}/*_fake.png')):
    name = os.path.basename(f).replace('_fake', '')
    fake, real = io.imread(f)[..., :3], io.imread(f'{real_dir}/{name}')[..., :3]
    rows.append(dict(id=name,
        psnr=psnr(real, fake, data_range=255),
        ssim=ssim(real, fake, channel_axis=2, data_range=255),
        deltaE=color.deltaE_ciede2000(color.rgb2lab(real), color.rgb2lab(fake)).mean(),
        lpips=loss_fn(to_t(real), to_t(fake)).item()))
df = pd.DataFrame(rows); df.to_csv(out_csv, index=False)
print(df.mean(numeric_only=True))

---
title: "Diffusion models 1"
slug: "diffusion-models-1"
date: 2026-03-09
draft: false
---

# Diffusion models

В генеративных моделях обычно существует компромисс между тремя свойствами:

- **High quality samples**
- **Fast sampling**
- **Mode coverage / diversity**

Разные семейства моделей оптимизируют разные части этого компромисса:

GAN — high quality + fast sampling  
VAE / flows — diversity + fast sampling  
Diffusion — high quality + diversity

Диффузионные модели развивают идеи **Denoising Autoencoders**. В них к данным добавляется шум, а модель обучается восстанавливать исходные данные. Такой подход обучает модель **двигать точки обратно к многообразию данных**, на котором лежит датасет.

Однако если пытаться сразу получить шум из изображения и обратно изображение из шума, процесс оказывается нестабильным. Поэтому вводится **итеративный процесс зашумления**, который затем обучаются обращать во времени.

# Forward diffusion process

Пусть изображение берётся из распределения данных

$$x_0 \sim p_{\text{data}}$$

Наша цель — постепенно превращать изображение в шум.

Можно было бы просто добавлять гауссовский шум на каждом шаге:

$$x_t = x_{t-1} + g(t)\epsilon_t, \quad \epsilon_t \sim \mathcal N(0, I)$$

Однако такой процесс (**variance exploding**) приводит к тому, что норма $x_t$ со временем растёт.

Чтобы контролировать рост дисперсии, используется схема, в которой часть изображения сохраняется, а часть заменяется шумом.

$$x_t = \sqrt{1-\beta_t}\, x_{t-1} + \sqrt{\beta_t}\,\epsilon_t,\quad \epsilon_t \sim \mathcal N(0,I)$$

Такой процесс называется **variance preserving diffusion**.

Из рекуррентной формулы можно напрямую выписать распределение перехода между соседними шагами:

$$q(x_t|x_{t-1}) = \mathcal N(x_t | \sqrt{1-\beta_t}\,x_{t-1}, \beta_t I)$$

Однако при обучении у нас есть только $x_0$. Поэтому необходимо получить распределение **$q(x_t|x_0)$**, которое позволит напрямую семплировать любой уровень шума.

Чтобы получить распределение $x_t$ через $x_0$, развернём рекурсию forward-процесса. После последовательной подстановки предыдущих шагов выражение для $x_t$ превращается в линейную комбинацию исходного изображения и гауссовских шумов.

$$x_t=\sqrt{1-\beta_t}x_{t-1}+\sqrt{\beta_t}\epsilon_t$$

$$x_t=\left(\prod_{s=1}^t \sqrt{1-\beta_s}\right)x_0+\sum_{s=1}^{t} c_s \epsilon_s$$

Так как это **линейная комбинация гауссовских случайных величин**, распределение $x_t$ также будет нормальным. Поэтому достаточно вычислить **математическое ожидание и дисперсию** чтобы определить параметры распределения.

Начнём с вычисления условного ожидания. Используем линейность ожидания и тот факт, что шум имеет нулевое среднее.

$$\mathbb{E}[x_t|x_{t-1}] = \sqrt{1-\beta_t}\mathbb{E}[x_{t-1}] + \sqrt{\beta_t}\mathbb{E}[\varepsilon_t]$$

Разворачивая рекурсию:

$$\mathbb{E}[x_t|x_0] = \left(\prod_{s=1}^{t}\sqrt{1-\beta_s}\right)\mathbb{E}[x_0] + \sum_{s=1}^{t}c_s\mathbb{E}[\epsilon_s]$$

$x_0$ — фиксированное изображение из датасета, а $\epsilon \sim \mathcal N(0,I)$, поэтому

$$\mathbb E[x_t|x_0] = \left(\prod_{s=1}^{t}\sqrt{1-\beta_s}\right)x_0$$

Введём обозначение и перепишем математическое ожидание с учетом новых обозначений

$$\alpha_t = \prod_{s=1}^{t}(1-\beta_s)$$


$$\mathbb E[x_t|x_0] = \sqrt{\alpha_t}x_0$$

Теперь вычислим дисперсию распределения $x_t$. Используем рекурсивную формулу дисперсии для линейной комбинации случайных величин.

$$\mathrm{Var}[x_t|x_{t-1}] = (1-\beta_t)\mathrm{Var}[x_{t-1}] + \beta_t\mathrm{Var}[\varepsilon_t]$$

Так как $\mathrm{Var}[\varepsilon_t] = I$:

$$\mathrm{Var}[x_t|x_{t-1}] = (1-\beta_t)\mathrm{Var}[x_{t-1}] + \beta_t I$$

Удобно переписать выражение через $I - \mathrm{Var}$ и затем развернуть рекурсию:

$$I-\mathrm{Var}[x_t|x_{t-1}] = (1-\beta_t)(I-\mathrm{Var}[x_{t-1}])$$

$$I-\mathrm{Var}[x_t] = \prod_{s=1}^{t}(1-\beta_s) I$$

Следовательно

$$\mathrm{Var}[x_t] = I-\prod_{s=1}^{t}(1-\beta_s)I$$

Используя обозначение $\alpha_t$:

$$\mathrm{Var}[x_t] = (1-\alpha_t)I$$

Теперь можно записать компактную формулу для итогового распределения, позволяющую напрямую семплировать $x_t$ из $x_0$:

$$q(x_t|x_0) = \mathcal N(x_t|\sqrt{\alpha_t}x_0,(1-\alpha_t)I)$$

Кроме того, мы хотим, чтобы после большого числа шагов изображение превращалось в стандартный гауссовский шум. Это выполняется, если

$$\alpha_T \approx 0,\quad q(x_T|x_0) \approx \mathcal N(0,I)$$

Используя формулу полной плотности:

$$q(x_T)=\int q(x_T|x_0)q(x_0)dx_0 \approx \mathcal N(0,I)$$

Таким образом **forward процесс превращает данные в стандартный гауссовский шум**.

# Reverse diffusion process

Теперь задача обратная: **восстановить изображение из шума**, двигаясь по шагам $T \rightarrow 0$.

Forward-процесс известен, но обратный переход

$$q(x_{t-1}|x_t)$$

оказывается сложным, поэтому его **аппроксимируют нейросетью**.

$$p_\theta(x_{t-1}|x_t)=\mathcal N(x_{t-1}|\mu_\theta(x_t),\sigma_t^2 I)$$

Процесс генерации изображения повторяет обратный диффузионный процесс.

$$x_T \sim \mathcal N(0,I),\quad t=T\dots1$$

$$x_{t-1}=\mu_\theta(x_t)+\sigma_t\epsilon$$

Полная вероятность прямого процесса:

$$q(x_0\dots x_T)=q(x_0)\prod_{t=1}^{T}q(x_t|x_{t-1})$$

Параметризованная обратная цепочка:

$$p_\theta(x_{0:T})=p(x_T)\prod_{t=1}^{T}p_\theta(x_{t-1}|x_t)$$

где начальное распределение

$$p(x_T)=\mathcal N(0,I)$$

# Обучение диффузионной модели

В процессе обучения мы хотим приблизиться наше исходное распределение к прямого процесса к обратному. Тогда обучение формулируется как минимизация KL-дивергенции между двумя вероятностными цепочками.

$$KL(q(x_{0},\dots,x_{T})||p^\theta(x_{0},\dots x_{T})) \to \min_{\theta}$$

или

$$\mathbb E_q\left[\log\frac{q(x_0,\dots x_{T})}{p^\theta(x_{0},\dots x_{T})}\right]$$

Раскрывая логарифм произведения, получаем

$$\mathbb E_q\left[\log q(x_0)+\sum\log q(x_t|x_{t-1})-\log p(x_T)-\sum\log p_\theta(x_{t-1}|x_t)\right]$$

Часть членов не зависит от $\theta$, поэтому они не влияют на оптимизацию. После преобразований objective принимает вид

$$\sum_{t=2}^{T}\mathbb E\left[KL(q(x_{t-1}|x_t,x_0)||p^\theta(x_{t-1}|x_t))\right]$$

Для вычисления KL необходимо знать распределение $q(x_{t-1}|x_t,x_0)$. Можно показать, что оно также является гауссовским.

$$q(x_{t-1}|x_t,x_0)=\mathcal N(x_{t-1}|A_t x_t + B_t x_0,C_t^2 I)$$

Нейросеть моделирует распределение

$$p^\theta(x_{t-1}|x_t)=\mathcal N(x_{t-1}|\mu^\theta(x_t),\sigma_t^2 I)$$

Если

$$p=\mathcal N(\mu_1,\sigma^2 I), \quad q=\mathcal N(\mu_2,\sigma^2 I)$$

то

$$KL(p||q)\propto||\mu_1-\mu_2||^2$$

Поэтому оптимизация KL сводится к минимизации квадратичной ошибки между средними.

Objective становится

$$\sum_t \mathbb E\left[||\mu_\theta(x_t)-(A_t x_t + B_t x_0)||^2\right]$$

Одна из возможных параметризаций:

$$\mu_\theta(x_t)=A_t x_t + B_t D_\theta(x_t)$$

где сеть предсказывает $x_0$.

Тогда loss принимает вид

$$\sum w_t||D_\theta(x_t)-x_0||^2$$

Используем формулу forward-процесса, связывающую зашумлённое изображение с исходным и шумом.

$$x_t=\sqrt{\alpha_t}x_0+\sqrt{1-\alpha_t}\epsilon$$

Выразим из неё $x_0$:

$$x_0=\frac{x_t-\sqrt{1-\alpha_t}\epsilon}{\sqrt{\alpha_t}}$$

Подставляя это выражение в objective, получаем задачу предсказания шума

$$\sum\mathbb E[||\epsilon_\theta(x_t,t)-\epsilon||^2]$$

Получили финальная функция потерь

$$L_{\text{simple}}=\mathbb E_{t,x_0,\epsilon}[\|\epsilon-\epsilon_\theta(\sqrt{\alpha_t}x_0+\sqrt{1-\alpha_t}\epsilon,t)\|^2]$$

Зная фунцию потерь можем записать шаг обучения. Пусть batch size равен $B$, тогда

1. $x_0^{(i)} \sim p_{data}$
2. $t^{(i)} \sim Uniform(1\dots T)$
3. $\epsilon^{(i)} \sim \mathcal N(0,I)$
4. $x_t=\sqrt{\alpha_t}x_0+\sqrt{1-\alpha_t}\epsilon$
5. $\frac1B\sum\|\epsilon_\theta(x_t,t)-\epsilon\|^2$

# Реализация

Некоторые полезные утилитные фукнции

```Python
def exists(x):
    return x is not None


def default(value, d):
    return value if exists(value) else d


def extract(a: torch.Tensor, t: torch.Tensor, x_shape):
    """
    Извлекает значения a[t] для каждого элемента батча и
    приводит форму к [B, 1, 1, 1, ...] для broadcasting.
    """
    batch_size = t.shape[0]
    out = a.gather(dim=0, index=t)
    return out.view(batch_size, *([1] * (len(x_shape) - 1)))
```

Кодируем номер шага $t$, чтобы было позиционное кодирование

```Python
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: [B] целые timesteps
        return: [B, dim]
        """
        half_dim = self.dim // 2
        device = t.device

        emb_scale = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        emb = t.float()[:, None] * emb[None, :]  # [B, half_dim]

        emb = torch.cat([emb.sin(), emb.cos()], dim=1)

        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))

        return emb
```

ResNet блок

```Python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.time_mlp = nn.Linear(time_dim, out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)

        time_term = self.time_mlp(t_emb)[:, :, None, None]
        h = h + time_term

        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)

        return h + self.shortcut(x)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x
```

Наша нейросеть которая будет предсказывать шум

```Python
class SimpleUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        channel_multipliers=(1, 2, 4),
        time_dim: int = 256,
    ):
        super().__init__()

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        c1 = base_channels * channel_multipliers[0]
        c2 = base_channels * channel_multipliers[1]
        c3 = base_channels * channel_multipliers[2]

        self.init_conv = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1)

        # Down
        self.down1 = ResidualBlock(c1, c1, time_dim)
        self.down2 = ResidualBlock(c1, c2, time_dim)
        self.pool1 = Downsample(c2)

        self.down3 = ResidualBlock(c2, c2, time_dim)
        self.down4 = ResidualBlock(c2, c3, time_dim)
        self.pool2 = Downsample(c3)

        # Middle
        self.mid1 = ResidualBlock(c3, c3, time_dim)
        self.mid2 = ResidualBlock(c3, c3, time_dim)

        # Up
        self.up1 = Upsample(c3)
        self.up_block1 = ResidualBlock(c3 + c3, c2, time_dim)
        self.up_block2 = ResidualBlock(c2, c2, time_dim)

        self.up2 = Upsample(c2)
        self.up_block3 = ResidualBlock(c2 + c2, c1, time_dim)
        self.up_block4 = ResidualBlock(c1, c1, time_dim)

        self.final_norm = nn.GroupNorm(8, c1)
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv2d(c1, in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embedding(t)

        x = self.init_conv(x)

        h1 = self.down1(x, t_emb)
        h2 = self.down2(h1, t_emb)
        x = self.pool1(h2)

        h3 = self.down3(x, t_emb)
        h4 = self.down4(h3, t_emb)
        x = self.pool2(h4)

        x = self.mid1(x, t_emb)
        x = self.mid2(x, t_emb)

        x = self.up1(x)
        x = torch.cat([x, h4], dim=1)
        x = self.up_block1(x, t_emb)
        x = self.up_block2(x, t_emb)

        x = self.up2(x)
        x = torch.cat([x, h2], dim=1)
        x = self.up_block3(x, t_emb)
        x = self.up_block4(x, t_emb)

        x = self.final_norm(x)
        x = self.final_act(x)
        x = self.final_conv(x)
        return x
```

Конфиг для модели (определяет шумы, которые будем добаволять, и размер картинки) и саму модель.

```Python
@dataclass
class DiffusionConfig:
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    image_size: int = 32
    channels: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class Diffusion(nn.Module):
    """
    Реализует:
    1. q(x_t | x_0)
    2. training loss для epsilon-prediction
    3. один шаг reverse sampling
    4. полное семплирование
    """
    def __init__(self, model: nn.Module, config: DiffusionConfig):
        super().__init__()
        self.model = model
        self.config = config
        self.num_timesteps = config.timesteps

        betas = torch.linspace(config.beta_start, config.beta_end, config.timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Регистрируем как buffers, чтобы они автоматически ездили на device
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recip_alphas",
            torch.sqrt(1.0 / alphas)
        )

        # posterior variance для q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)

    # --------------------------------------------------------
    # Forward process
    # --------------------------------------------------------

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise=None) -> torch.Tensor:
        """
        Семплирует x_t из q(x_t | x_0):
            x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
        """
        noise = default(noise, torch.randn_like(x_start))

        sqrt_alpha_bar_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha_bar_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alpha_bar_t * x_start + sqrt_one_minus_alpha_bar_t * noise

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    def loss(self, x_start: torch.Tensor) -> torch.Tensor:
        """
        1) sample x_0
        2) sample t ~ Uniform({1, ..., T})
        3) sample eps ~ N(0, I)
        4) construct x_t
        5) minimize ||eps_theta(x_t, t) - eps||^2
        """
        batch_size = x_start.shape[0]
        device = x_start.device

        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device)
        noise = torch.randn_like(x_start)

        x_t = self.q_sample(x_start, t, noise)
        noise_pred = self.model(x_t, t)

        return F.mse_loss(noise_pred, noise)

    # --------------------------------------------------------
    # Reverse process
    # --------------------------------------------------------

    @torch.no_grad()
    def p_mean(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Если модель предсказывает eps_theta(x_t, t), то
            mu_theta(x_t, t) =
                1 / sqrt(alpha_t) *
                (
                    x_t - beta_t / sqrt(1 - alpha_bar_t) * eps_theta(x_t, t)
                )
        """
        beta_t = extract(self.betas, t, x_t.shape)
        sqrt_one_minus_alpha_bar_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )
        sqrt_recip_alpha_t = extract(self.sqrt_recip_alphas, t, x_t.shape)

        eps_theta = self.model(x_t, t)

        mean = sqrt_recip_alpha_t * (
            x_t - beta_t / sqrt_one_minus_alpha_bar_t * eps_theta
        )
        return mean

    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Один reverse step:
            x_{t-1} = mu_theta(x_t, t) + sigma_t * z,   z ~ N(0, I)
        На последнем шаге шум не добавляем.
        """
        mean = self.p_mean(x_t, t)

        posterior_variance_t = extract(self.posterior_variance, t, x_t.shape)

        # если t == 0, это последний шаг, шум не нужен
        nonzero_mask = (t != 0).float().view(x_t.shape[0], *([1] * (x_t.ndim - 1)))
        noise = torch.randn_like(x_t)

        return mean + nonzero_mask * torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, batch_size: int) -> torch.Tensor:
        """
        Полный reverse sampling:
            x_T ~ N(0, I)
            for t = T-1, ..., 0:
                x_{t} <- p_sample(...)
        """
        device = self.betas.device
        shape = (
            batch_size,
            self.config.channels,
            self.config.image_size,
            self.config.image_size,
        )

        x = torch.randn(shape, device=device)

        for timestep in reversed(range(self.num_timesteps)):
            t = torch.full((batch_size,), timestep, device=device, dtype=torch.long)
            x = self.p_sample(x, t)

        return x

    @torch.no_grad()
    def sample_trajectory(self, batch_size: int, every: int = 100):
        """
        Возвращает промежуточные состояния reverse-процесса.
        Полезно для конспекта и визуализации.
        """
        device = self.betas.device
        shape = (
            batch_size,
            self.config.channels,
            self.config.image_size,
            self.config.image_size,
        )

        x = torch.randn(shape, device=device)
        trajectory = [x.detach().cpu()]

        for timestep in reversed(range(self.num_timesteps)):
            t = torch.full((batch_size,), timestep, device=device, dtype=torch.long)
            x = self.p_sample(x, t)

            if timestep % every == 0 or timestep == 0:
                trajectory.append(x.detach().cpu())

        return trajectory
```

```Python
def train_one_epoch(
    diffusion: Diffusion,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device: str,
    grad_clip: float | None = 1.0,
):
    diffusion.train()
    total_loss = 0.0

    for x, _ in dataloader:
        x = x.to(device)

        optimizer.zero_grad()
        loss = diffusion.loss(x)
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), grad_clip)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)
```

# Ссылки

- [arXiv:2006.11239 - Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [Denoising Diffusion Models: A Generative Learning Big Bang](https://cvpr2023-tutorial-diffusion-models.github.io/)
- [DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://dreambooth.github.io/)

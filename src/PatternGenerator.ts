/**
 * Creates a colorful gradient background with three color stops
 */
function createGradient(
  ctx: OffscreenCanvasRenderingContext2D,
  height: number,
  color1?: string,
  color2?: string,
  color3?: string,
): CanvasGradient {
  const gradient = ctx.createLinearGradient(0, 0, 0, height);

  if (color1 && color2 && color3) {
    // Use provided colors
    gradient.addColorStop(0, color1);
    gradient.addColorStop(0.5, color2);
    gradient.addColorStop(1, color3);
  } else {
    // Generate random colors
    const hue = Math.random() * 360;
    const randomColor1 = `hsl(${hue}, ${80 + Math.random() * 20}%, 50%)`;
    const randomColor2 = `hsl(${(hue + 120) % 360}, ${80 + Math.random() * 20}%, 90%)`;
    const randomColor3 = `hsl(${(hue + 240) % 360}, ${80 + Math.random() * 20}%, 50%)`;

    gradient.addColorStop(0, randomColor1);
    gradient.addColorStop(0.5, randomColor2);
    gradient.addColorStop(1, randomColor3);
  }

  return gradient;
}

/**
 * Runs the same draw commands for the original position and all wrapped
 * positions
 */
function wrapped(
  ctx: OffscreenCanvasRenderingContext2D,
  width: number,
  height: number,
  drawFn: (ctx: OffscreenCanvasRenderingContext2D) => void,
) {
  // Original position
  drawFn(ctx);

  // Left edge wrap
  ctx.save();
  ctx.translate(width, 0);
  drawFn(ctx);
  ctx.restore();

  // Right edge wrap
  ctx.save();
  ctx.translate(-width, 0);
  drawFn(ctx);
  ctx.restore();

  // Top edge wrap
  ctx.save();
  ctx.translate(0, height);
  drawFn(ctx);
  ctx.restore();

  // Bottom edge wrap
  ctx.save();
  ctx.translate(0, -height);
  drawFn(ctx);
  ctx.restore();

  // Corner wraps
  ctx.save();
  ctx.translate(width, height);
  drawFn(ctx);
  ctx.restore();

  ctx.save();
  ctx.translate(-width, height);
  drawFn(ctx);
  ctx.restore();

  ctx.save();
  ctx.translate(width, -height);
  drawFn(ctx);
  ctx.restore();

  ctx.save();
  ctx.translate(-width, -height);
  drawFn(ctx);
  ctx.restore();
}

/**
 * Generates a noise pattern with scaled-up random grayscale noise and colorful
 * gradient overlay
 */
export function generateNoisePattern(
  gradientColor1?: string,
  gradientColor2?: string,
  gradientColor3?: string,
): HTMLCanvasElement {
  const width = 256;
  const height = 1024;
  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext('2d')!;

  // Create grayscale noise first
  const noiseSize = 8; // Size of each noise "pixel"
  const noiseWidth = Math.ceil(width / noiseSize);
  const noiseHeight = Math.ceil(height / noiseSize);

  // Generate random grayscale noise
  for (let y = 0; y < noiseHeight; y++) {
    for (let x = 0; x < noiseWidth; x++) {
      const noiseValue = Math.random() * 0.7;
      const gray = Math.floor(noiseValue * 255);

      ctx.fillStyle = `rgb(${gray}, ${gray}, ${gray})`;
      ctx.fillRect(x * noiseSize, y * noiseSize, noiseSize, noiseSize);
    }
  }

  // Overlay colorful gradient using blend mode
  ctx.globalCompositeOperation = 'overlay';
  ctx.fillStyle = createGradient(
    ctx,
    height,
    gradientColor1,
    gradientColor2,
    gradientColor3,
  );
  ctx.fillRect(0, 0, width, height);
  ctx.globalCompositeOperation = 'source-over';

  return canvas as unknown as HTMLCanvasElement;
}

/**
 * Generates a confetti pattern with random colored circles
 */
export function generateConfettiPattern(
  gradientColor1?: string,
  gradientColor2?: string,
  gradientColor3?: string,
): HTMLCanvasElement {
  const width = 256;
  const height = 1024;
  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext('2d')!;

  // Check if user prefers dark scheme
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;

  // Create colorful gradient background with three random stops
  ctx.fillStyle = createGradient(
    ctx,
    height,
    gradientColor1,
    gradientColor2,
    gradientColor3,
  );
  ctx.fillRect(0, 0, width, height);

  // Generate ~600 random colored circles (doubled from 300)
  for (let i = 0; i < 900; i++) {
    const x = Math.random() * width;
    const y = Math.random() * height;
    const radius = 4 + Math.random() * 16; // 2-18px radius
    const hue = Math.random() * 360;
    const saturation = 50 + Math.random() * 50; // 50-100%
    const lightness = 40 + Math.random() * 40; // 40-80%

    // Draw shadow first (darker, slightly offset)
    const shadowOffset = 2;
    const shadowColor = prefersDark
      ? 'rgba(255, 255, 255, 0.1)'
      : 'rgba(0, 0, 0, 0.2)';

    wrapped(ctx, width, height, (ctx) => {
      ctx.fillStyle = shadowColor;
      ctx.beginPath();
      ctx.arc(x + shadowOffset, y + shadowOffset, radius, 0, 2 * Math.PI);
      ctx.fill();
    });

    // Draw the main circle
    wrapped(ctx, width, height, (ctx) => {
      ctx.fillStyle = `hsl(${hue}, ${saturation}%, ${lightness}%)`;
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, 2 * Math.PI);
      ctx.fill();
    });
  }

  return canvas as unknown as HTMLCanvasElement;
}

/**
 * Generates a sprinkles pattern with random colored lines with circular end
 * caps
 */
export function generateSprinklesPattern(
  gradientColor1?: string,
  gradientColor2?: string,
  gradientColor3?: string,
): HTMLCanvasElement {
  const width = 256;
  const height = 1024;
  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext('2d')!;

  // Check if user prefers dark scheme
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;

  // Create colorful gradient background with three random stops
  ctx.fillStyle = createGradient(
    ctx,
    height,
    gradientColor1,
    gradientColor2,
    gradientColor3,
  );
  ctx.fillRect(0, 0, width, height);

  for (let i = 0; i < 900; i++) {
    const x1 = Math.random() * width;
    const y1 = Math.random() * height;
    const length = 8 + Math.random() * 16;
    const angle = Math.random() * 2 * Math.PI;
    const x2 = x1 + Math.cos(angle) * length;
    const y2 = y1 + Math.sin(angle) * length;
    const thickness = (1.5 + Math.random() * 4.5) * 1.5;
    const hue = Math.random() * 360;
    const saturation = 50 + Math.random() * 50;
    const lightness = 40 + Math.random() * 40;

    // Draw shadow first (darker, slightly offset)
    const shadowOffset = 2;
    const shadowColor = prefersDark
      ? 'rgba(255, 255, 255, 0.1)'
      : 'rgba(0, 0, 0, 0.2)';

    wrapped(ctx, width, height, (ctx) => {
      ctx.strokeStyle = shadowColor;
      ctx.lineWidth = thickness;
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.moveTo(x1 + shadowOffset, y1 + shadowOffset);
      ctx.lineTo(x2 + shadowOffset, y2 + shadowOffset);
      ctx.stroke();
    });

    // Draw the main line
    wrapped(ctx, width, height, (ctx) => {
      ctx.strokeStyle = `hsl(${hue}, ${saturation}%, ${lightness}%)`;
      ctx.lineWidth = thickness;
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    });
  }

  return canvas as unknown as HTMLCanvasElement;
}

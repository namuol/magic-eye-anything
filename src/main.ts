import {
  DepthEstimationPipelineOutput,
  pipeline,
  RawImage,
} from '@huggingface/transformers';

import {PixelGrid} from './PixelGrid';

async function main() {
  hide('preloader');

  const hasWebGPU = navigator.gpu !== undefined;

  const hasFp16 =
    hasWebGPU &&
    (async () => {
      try {
        const adapter = await navigator.gpu.requestAdapter();
        return adapter!.features.has('shader-f16');
      } catch {
        return false;
      }
    })();

  show('loader');
  const depthEstimator = await pipeline(
    'depth-estimation',
    'onnx-community/depth-anything-v2-small',
    {
      dtype: hasFp16 ? 'fp16' : undefined,
      device: hasWebGPU ? 'webgpu' : undefined,
    },
  );
  hide('loader');

  show('image-chooser');

  // Wait for the user to choose a photo...
  const imageChooser = document.getElementById(
    'choose-a-photo',
  ) as HTMLInputElement;

  // When the user chooses a photo, load it into the canvas
  imageChooser.addEventListener('change', async (e) => {
    const file = (e.target as HTMLInputElement).files?.[0];
    if (!file) return;

    hide('image-chooser');
    show('loading-depth-estimation');

    const image = await RawImage.fromBlob(file);
    const {depth} = (await depthEstimator(
      image,
    )) as DepthEstimationPipelineOutput;
    const canvasElement = document.getElementById(
      'canvas',
    ) as HTMLCanvasElement;
    const hiddenImageCanvas = new OffscreenCanvas(
      canvasElement.width,
      canvasElement.height,
    );
    const hiddenImageCtx = hiddenImageCanvas.getContext('2d')!;
    {
      hiddenImageCtx.fillStyle = 'black';
      hiddenImageCtx.fillRect(
        0,
        0,
        hiddenImageCanvas.width,
        hiddenImageCanvas.height,
      );
      // Draw the image centered on the canvas
      drawImageCentered(depth.toCanvas(), hiddenImageCanvas);
    }

    const hiddenImage = new PixelGrid(
      hiddenImageCtx.getImageData(
        0,
        0,
        hiddenImageCanvas.width,
        hiddenImageCanvas.height,
      ),
    );
    const minDisparity = Math.floor(hiddenImageCanvas.width * 0.15);
    const maxDisparity = Math.floor(hiddenImageCanvas.width * 0.2);
    const disparityScale = 1;
    const tileWidth = minDisparity;
    const patternImage = (await RawImage.fromURL('emoji.png')).toCanvas();
    const tileHeight = (tileWidth * patternImage.height) / patternImage.width;
    const patternCanvas = new OffscreenCanvas(tileWidth, tileHeight);
    fillImage(patternImage, patternCanvas, tileWidth);

    const pattern = new PixelGrid(
      patternCanvas
        .getContext('2d')!
        .getImageData(0, 0, patternCanvas.width, patternCanvas.height),
    );

    const outputCanvas = new OffscreenCanvas(
      hiddenImageCanvas.width,
      hiddenImageCanvas.height,
    );
    const output = new PixelGrid(
      outputCanvas
        .getContext('2d')!
        .getImageData(0, 0, outputCanvas.width, outputCanvas.height),
    );

    for (let y = 0; y < output.height; ++y) {
      for (let x = 0; x < output.width; ++x) {
        const disparity = hiddenImage.get(x, y)[0] / 255;
        const offset = Math.floor(
          disparity * (maxDisparity - minDisparity) * disparityScale,
        );
        if (x < minDisparity) {
          output.set(x, y, pattern.get((x + offset) % minDisparity, y));
        } else {
          output.set(x, y, output.get(x + offset - minDisparity, y));
        }
      }
    }

    const canvas = document.getElementById('canvas') as HTMLCanvasElement;
    const ctx = canvas.getContext('2d')!;
    ctx.putImageData(output.imageData, 0, 0);

    hide('messages');
    show('canvas');
    show('save-image-container');

    // Add save image functionality
    const saveButton = document.getElementById(
      'save-image',
    ) as HTMLButtonElement;
    saveButton.addEventListener('click', () => {
      const link = document.createElement('a');
      link.download = 'autostereogram.png';
      link.href = canvas.toDataURL();
      link.click();
    });
  });
}

main();

/**
 * Draws an image centered on a canvas with the correct aspect ratio, fitting
 * the entire picture.
 */
function drawImageCentered(
  /** The RawImage to draw */
  image: HTMLCanvasElement | OffscreenCanvas,
  /** The canvas to draw on */
  canvas: HTMLCanvasElement | OffscreenCanvas,
) {
  const ctx = canvas.getContext('2d') as CanvasRenderingContext2D;

  // Get dimensions
  const imageWidth = image.width;
  const imageHeight = image.height;
  const canvasWidth = canvas.width;
  const canvasHeight = canvas.height;

  // Calculate scale to fit the entire image
  const scaleX = canvasWidth / imageWidth;
  const scaleY = canvasHeight / imageHeight;
  const scale = Math.min(scaleX, scaleY);

  // Calculate centered position
  const scaledWidth = imageWidth * scale;
  const scaledHeight = imageHeight * scale;
  const x = (canvasWidth - scaledWidth) / 2;
  const y = (canvasHeight - scaledHeight) / 2;

  ctx.drawImage(image, x, y, scaledWidth, scaledHeight);
}

/**
 * Fills the canvas with the image, as a repeating pattern
 */
function fillImage(
  image: HTMLCanvasElement | OffscreenCanvas,
  canvas: HTMLCanvasElement | OffscreenCanvas,
  /**
   * The width of the tiles to use for the pattern; the height will be
   * calculated to maintain the aspect ratio.
   */
  tileWidth: number,
) {
  const ctx = canvas.getContext('2d') as
    | CanvasRenderingContext2D
    | OffscreenCanvasRenderingContext2D;

  // Calculate tile height to maintain aspect ratio
  const tileHeight = (tileWidth * image.height) / image.width;

  // Calculate how many tiles we need to fill the canvas
  const tilesX = Math.ceil(canvas.width / tileWidth);
  const tilesY = Math.ceil(canvas.height / tileHeight);

  // Draw the image as a repeating pattern
  for (let y = 0; y < tilesY; y++) {
    for (let x = 0; x < tilesX; x++) {
      ctx.drawImage(
        image,
        x * tileWidth,
        y * tileHeight,
        tileWidth,
        tileHeight,
      );
    }
  }
}

type UiElementId =
  | 'messages'
  | 'title'
  | 'preloader'
  | 'loader'
  | 'warning'
  | 'run-demo'
  | 'exit-demo'
  | 'image-chooser'
  | 'loading-depth-estimation'
  | 'canvas'
  | 'save-image-container';

function hide(id: UiElementId) {
  document.getElementById(id)!.hidden = true;
}

function show(id: UiElementId) {
  document.getElementById(id)!.hidden = false;
}

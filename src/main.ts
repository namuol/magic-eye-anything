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
    const hiddenImageCanvas = new OffscreenCanvas(1280, 720);
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

    const noiseCanvas = new OffscreenCanvas(
      hiddenImageCanvas.width,
      hiddenImageCanvas.height,
    );
    const noise = new PixelGrid(
      noiseCanvas
        .getContext('2d')!
        .getImageData(0, 0, noiseCanvas.width, noiseCanvas.height),
    );

    function makeNoise() {
      for (let y = 0; y < noise.height; ++y) {
        for (let x = 0; x < noise.width; ++x) {
          // const c = Math.random() < 0.5 ? 255 : 0;
          //
          // const c = Math.floor(Math.random() * 256); noise.set(x, y, [c, c,
          // c, 255]);

          noise.set(x, y, [
            Math.floor(Math.random() * 256),
            Math.floor(Math.random() * 256),
            Math.floor(Math.random() * 256),
            255,
          ]);
        }
      }
    }
    makeNoise();

    const outputCanvas = new OffscreenCanvas(
      hiddenImageCanvas.width,
      hiddenImageCanvas.height,
    );
    const output = new PixelGrid(
      outputCanvas
        .getContext('2d')!
        .getImageData(0, 0, outputCanvas.width, outputCanvas.height),
    );
    const minDisparity = Math.floor(hiddenImageCanvas.width * 0.15);
    const maxDisparity = Math.floor(hiddenImageCanvas.width * 0.2);

    for (let y = 0; y < output.height; ++y) {
      for (let x = 0; x < output.width; ++x) {
        const disparity = hiddenImage.get(x, y)[0] / 255;
        const offset = Math.floor(disparity * (maxDisparity - minDisparity));
        if (x < minDisparity) {
          output.set(x, y, noise.get((x + offset) % minDisparity, y));
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
  const ctx = canvas.getContext('2d')!;

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
  | 'canvas';

function hide(id: UiElementId) {
  document.getElementById(id)!.hidden = true;
}

function show(id: UiElementId) {
  document.getElementById(id)!.hidden = false;
}

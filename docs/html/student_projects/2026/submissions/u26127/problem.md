
* _this is an example file you should finish_
* write problem description, how the algorithm/baseline code works, and where you think you can apply parallel computing
* if you are looking at this file from Jupyter, you can see the preview by right-clicking on the cell and "Show Markdown Preview"

# Image stitching from scratch

(Baseline code not done yet, so this assignment is not finished. When done, I will re-submit from Jupyter as well as in UTOL.)

## Problem

Image stitching is the process of combining multiple images/photos, with overlapping views of the scene,
into a larger composed image with a wider view of the scene.
The result is usually a segmented panorama image or one with full 360° view.
Furthermore, rather than simply warping random image mosaics, appearance features on edges are aligned and
"stitched" on the output panorama, ideally without noticeable visual seams.

The problem solved here is unlike the panorama feature commonly seen in smartphone camera apps:

1. There's no real-time stream of images, but to stitch from sparse samples with less overlaps of views.
2. Lack of continuous stream also means image samples are unordered without associations in terms of overlaps.
3. No IMU sensor to potentially aid with tracking camera orientation.

To ease the problem, I will begin with relaxing 2., requiring overlaps in input images.
This will force the user to provide input images in a reasonable order.

Additional assumptions:
* Input images are taken by the same camera with same settings (or at least similar).
* We may ignore fine-grained improvements and optimization over the appearance of the image.
  Those extra credits will only be considered if time really permits.

## How the algorithm/baseline code works

Since the task is rather high-level, I would omit the exact algorithms used for each step in the pipeline.
Roughly speaking, for each image we have `feature detection -> matching -> solve for transformation`.
Once all transformations are resolved, the scene can be built as some environment representation and/or
rendered into a final panoramic image.

(Python-like pseudocode below; may contain bugs...)

```python
def stitch_images(src_images: Iterable[Image]) -> ImageMosaics:
  panorama = ImageMosaics()

  features = []
  last_image_idx = -1
  for i, image in enumerate(src_images):

    features.append(detect_features(image))  # e.g. SIFT, ORB

    # For the baseline, assumes overlaps of views between adjacent source images)
    if last_image_idx < 0:
      continue
    matches = match_features(features[i], features[last_image_idx])

    if not good_matches(matches):
      continue  # simply discards current image

    points_curr, points_last = matches.src_points, matches.dst_points
    transform = solve_relative_transformation(points_curr, points_last)
    # e.g. relative rotation, homography

    # optional (optimizing transforms, reduce discontinuities, etc.)
    other_local_routines(...)

    panorama.add(image, transform)
    last_image_idx = i

  other_global_routines(...)  # optional

  return panorama


# Explicitly treat rendering as a separate routine.
# Other kinds of rendering are possible.
def render_to_equirectangular(panorama: Panorama, output_height: int) -> Image:
  output = Image(size=(output_height*2, output_height))
  for (image, transform) in Panorama:
    warp_to_equirectangular(output, image, transform)
  return output


def main():
  image_loader = ImageLoader("image/dir")
  panorama = stitch_images(image_loader)
  write_image("output/dir", render_to_equirectangular(panorama))

```

## Where is parallelism

* In general, most steps of the stitching pipeline operate independently on different images or pairs of images.
* For feature detection on a single image, a thread-based mechanism may be too heavy, so I'm thinking about SIMD.
* When having loads of input images, feature detection may be divided up into groups of images.
  A multithreaded approach could be beneficial.
* The matching step only reads existing feature descriptors. The naive brute-force matching can be parallelized
  simply by splitting the comparison work by feature points. The better FLANN-based matching also have some potential
  that I have not yet digged into.
* Solving a reasonable transformation between images, often a homography, can involve Direct Linear
  Transformormation (DLT), which requires solving systems of linear equations. This would call for some
  parallelized matrix operations.
* RANSAC is commonly used to reject outlier matches. The iterations are independent and evaluations on
  data points can also be split.
* Rendering to a panoramic view/canvas can be performed in parallel once the tranformations of each image
  is determined. (although I believe it is also feasible to render by sequential calls to drawing with OpenGL
  when available.)

I will prioritize parallelism on CPU, as the application will be more portable across types of devices,
especially those on the edge.
GPU-offloading may be considered when appropriate.
As an extra bonus, it's also worth exploring on-device processing of the pipeline on GPU with CUDA.

## Personal remarks

* My focus for this course: to get some hands-on experience implementing parallelized computation.
  Here are some principles that led to choosing this problem:
  * No ambition for novelty. A known classical algorithm is better.
  * A practically useful application: something I can put into actual use and have some fun,
    and stays open for extension and polish even after the course ends.
  * Avoid too much irrelevant effort to get the program to work.
    This rules out learning-based programs like DNNs that requires dataset preparation,
    training steps, and experiments.
* I've also considered solving basic *structure from motion*, which has some useful downstream
  applications in photogrammetry and 3D reconstruction.
  That said, the result requires 3D visualization, which I think is harder to integrate to a
  JupyterLab-based workflow. (probably need a Jupyter extension)
  I'm still partially open to this problem, though, and it shares some subroutines with image stitching.

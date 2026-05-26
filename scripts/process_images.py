#!/usr/bin/env python3
"""
Batch image processing script.

Features:
1. Read all images in a given folder.
2. Resize images proportionally to a target pixel-count range.
   - Max pixels: 3211264
   - Min pixels: 4*28*28 = 3136
3. Save processed images (optionally overwriting the originals or writing to a new directory).
"""

import os
import argparse
import math
from pathlib import Path
from PIL import Image
from PIL.Image import Image as ImageObject
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
Image.MAX_IMAGE_PIXELS = None  # Matches the setting used in the training scripts
# Supported image formats
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

# Pixel limits
MAX_PIXELS = 3211264
MIN_PIXELS = 4 * 28 * 28  # 3136


def resize_image(image: ImageObject, max_pixels: int, min_pixels: int) -> ImageObject:
    """
    Resize an image proportionally so that its pixel count falls within the target range.

    Args:
        image: PIL Image object.
        max_pixels: Maximum pixel count.
        min_pixels: Minimum pixel count.

    Returns:
        Processed PIL Image object.
    """
    width, height = image.width, image.height
    current_pixels = width * height

    # If the image is too small, upscale it
    if current_pixels < min_pixels:
        resize_factor = math.sqrt(min_pixels / current_pixels)
        new_width = int(width * resize_factor)
        new_height = int(height * resize_factor)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        current_pixels = new_width * new_height

    # If the image is too large, downscale it
    if current_pixels > max_pixels:
        resize_factor = math.sqrt(max_pixels / current_pixels)
        new_width = int(width * resize_factor)
        new_height = int(height * resize_factor)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return image


def process_single_image(input_path: Path, output_path: Path, max_pixels: int, min_pixels: int) -> tuple[bool, str, bool]:
    """
    Process a single image.

    Args:
        input_path: Input image path.
        output_path: Output image path.
        max_pixels: Maximum pixel count.
        min_pixels: Minimum pixel count.

    Returns:
        (success, message, whether the file was actually processed).
    """
    try:
        # Open the image
        with Image.open(input_path) as img:
            original_pixels = img.width * img.height

            # Check whether the image needs processing
            needs_resize = False
            needs_convert = img.mode != 'RGB'

            if original_pixels < min_pixels:
                needs_resize = True
            elif original_pixels > max_pixels:
                needs_resize = True

            # If neither resize nor conversion is needed, skip it
            if not needs_resize and not needs_convert:
                return True, f"SKIP: {original_pixels} pixels ({img.width}x{img.height}) - already in range", False

            # Needs processing: convert format or resize
            if needs_convert:
                img = img.convert('RGB')

            # Resize if needed
            if needs_resize:
                resized_img = resize_image(img, max_pixels, min_pixels)
            else:
                resized_img = img

            # Save the image
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Preserve the original format; use high quality when saving JPEG
            if input_path.suffix.lower() in ['.jpg', '.jpeg']:
                resized_img.save(output_path, 'JPEG', quality=95, optimize=True)
            else:
                resized_img.save(output_path, quality=95, optimize=True)
            
            new_pixels = resized_img.width * resized_img.height
            
            if needs_resize:
                return True, f"RESIZE: {original_pixels} -> {new_pixels} pixels ({resized_img.width}x{resized_img.height})", True
            else:
                return True, f"CONVERT: {original_pixels} pixels ({resized_img.width}x{resized_img.height}) - RGB conversion only", True
    
    except Exception as e:
        return False, f"Error: {str(e)}", False


def get_image_files(directory: Path) -> list[Path]:
    """
    Return all image files in a directory.

    Args:
        directory: Directory path.

    Returns:
        List of image file paths.
    """
    image_files = []
    for ext in SUPPORTED_FORMATS:
        image_files.extend(directory.rglob(f'*{ext}'))
        image_files.extend(directory.rglob(f'*{ext.upper()}'))
    
    return sorted(image_files)


def main():
    parser = argparse.ArgumentParser(description="Batch resize images proportionally into a target pixel-count range")
    parser.add_argument("--input_dir", type=str, required=True, help="Input image directory")
    parser.add_argument("--output_dir", type=str, default=None, help="Output image directory (overwrites originals by default)")
    parser.add_argument("--max_pixels", type=int, default=MAX_PIXELS, help=f"Maximum pixel count (default: {MAX_PIXELS})")
    parser.add_argument("--min_pixels", type=int, default=MIN_PIXELS, help=f"Minimum pixel count (default: {MIN_PIXELS})")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of worker threads (default: 16)")
    parser.add_argument("--dry_run", action="store_true", help="Only print files to be processed; don't actually process them")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: input directory does not exist: {input_dir}")
        return

    # Determine the output directory
    if args.output_dir is None:
        output_dir = input_dir  # Overwrite originals
        print(f"Output directory: {input_dir} (will overwrite originals)")
    else:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir} (saving to a new directory)")

    # Collect image files
    print(f"\nScanning image files...")
    image_files = get_image_files(input_dir)

    if not image_files:
        print("No image files found")
        return

    print(f"Found {len(image_files)} image files")

    if args.dry_run:
        print("\n[DRY RUN] Files that would be processed:")
        for img_file in image_files[:10]:  # Show only the first 10
            print(f"  {img_file}")
        if len(image_files) > 10:
            print(f"  ... and {len(image_files) - 10} more files")
        return

    # Process images
    print(f"\nStarting image processing (max pixels: {args.max_pixels}, min pixels: {args.min_pixels})...")
    
    success_count = 0
    error_count = 0
    skipped_count = 0
    processed_count = 0
    total_original_size = 0
    total_new_size = 0
    
    def process_with_path(img_file: Path):
        """Wrapper that processes a single image file."""
        if args.output_dir is None:
            # Overwrite the original file
            output_path = img_file
        else:
            # Preserve the relative path structure
            relative_path = img_file.relative_to(input_dir)
            output_path = output_dir / relative_path

        return process_single_image(img_file, output_path, args.max_pixels, args.min_pixels)

    # Process in parallel with a thread pool
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(process_with_path, img_file): img_file
            for img_file in image_files
        }

        # Collect results
        for future in tqdm(as_completed(future_to_file), total=len(image_files), desc="Processing images"):
            img_file = future_to_file[future]
            try:
                success, message, was_processed = future.result()
                if success:
                    success_count += 1
                    if was_processed:
                        processed_count += 1
                        # Parse pixel info for statistics
                        if "pixels" in message:
                            try:
                                if "->" in message:
                                    parts = message.split("->")
                                    if len(parts) == 2:
                                        original_pixels = int(parts[0].split()[-1])
                                        new_pixels = int(parts[1].split()[0])
                                        total_original_size += original_pixels
                                        total_new_size += new_pixels
                                elif "CONVERT" in message:
                                    # Format-conversion-only case
                                    pixels = int(message.split()[1])
                                    total_original_size += pixels
                                    total_new_size += pixels
                            except:
                                pass
                    else:
                        skipped_count += 1
                else:
                    error_count += 1
                    print(f"\n[ERROR] {img_file.name}: {message}")
            except Exception as e:
                error_count += 1
                print(f"\n[EXCEPTION] {img_file.name}: {str(e)}")

    # Print statistics
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    print(f"  Total: {len(image_files)} files")
    print(f"  Processed: {processed_count} files (resize or format conversion required)")
    print(f"  Skipped: {skipped_count} files (already within range, no processing needed)")
    print(f"  Succeeded: {success_count} files")
    print(f"  Failed: {error_count} files")

    if processed_count > 0 and total_original_size > 0 and total_new_size > 0:
        avg_original = total_original_size / processed_count
        avg_new = total_new_size / processed_count
        reduction = (1 - total_new_size / total_original_size) * 100 if total_original_size > 0 else 0
        print(f"\n  Stats for processed files:")
        print(f"    Average original pixels: {avg_original:.0f}")
        print(f"    Average processed pixels: {avg_new:.0f}")
        if reduction > 0:
            print(f"    Average reduction: {reduction:.1f}%")
        elif reduction < 0:
            print(f"    Average increase: {abs(reduction):.1f}%")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    DATASET_DIR = os.path.join(BASE_DIR, "Dataset", "input")
    OUTPUT_DIR = os.path.join(BASE_DIR, "Dataset", "output")

    RESIZE_TO = (256, 256)
    USE_SKULL_STRIP = True
    USE_PARALLEL = True  # Set to False for debugging

    # Create output folder if not exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Get image paths (only PNGs as per your structure)
    image_paths = []
    for file in os.listdir(DATASET_DIR):
        if file.lower().endswith(".png"):
            image_paths.append(os.path.join(DATASET_DIR, file))

    if len(image_paths) == 0:
        raise FileNotFoundError(f"No PNG images found in {DATASET_DIR}")

    print(f"Found {len(image_paths)} images")

    # Process all images
    results = batch_process_images(
        image_paths,
        resize_dim=RESIZE_TO,
        use_skull_strip=USE_SKULL_STRIP,
        parallel=USE_PARALLEL,
        max_workers=4
    )

    # Display + SAVE results
    if len(results) == 0:
        print("No results to display")
    else:
        num_images = len(results)
        fig, axes = plt.subplots(num_images, 4, figsize=(16, 4 * num_images))
        if num_images == 1:
            axes = axes.reshape(1, -1)
        
        for i, (orig, mask, overlay, metadata, filename) in enumerate(results):
            name = os.path.splitext(filename)[0]

            # ✅ SAVE OUTPUTS
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_orig.png"), cv2.cvtColor(orig, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_mask.png"), (mask * 255).astype(np.uint8))
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            # Original
            axes[i, 0].imshow(orig)
            axes[i, 0].set_title(f'{filename}\nOriginal')
            axes[i, 0].axis('off')
            
            # Mask
            axes[i, 1].imshow(mask, cmap='gray')
            axes[i, 1].set_title(f'Tumor Mask\nArea: {metadata["tumor_area"]}px')
            axes[i, 1].axis('off')
            
            # Overlay
            axes[i, 2].imshow(overlay)
            axes[i, 2].set_title(f'Overlay\nTime: {metadata["processing_time"]:.3f}s')
            axes[i, 2].axis('off')
            
            # Stats
            axes[i, 3].axis('off')
            stats_text = (
                f"Candidates: {metadata['num_candidates']}\n"
                f"Mean Intensity: {metadata['mean_intensity']:.1f}\n"
                f"Processing: {metadata['processing_time']:.3f}s"
            )
            axes[i, 3].text(0.1, 0.5, stats_text, fontsize=10, 
                           verticalalignment='center')
        
        plt.tight_layout()
        plt.show()
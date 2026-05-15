/**
 * M3DHP Brain MRI Segmentation – Frontend Logic
 */
(function () {
  "use strict";

  // DOM references
  const dropZone      = document.getElementById("drop-zone");
  const fileInput     = document.getElementById("file-input");
  const previewWrap   = document.getElementById("preview-container");
  const previewImg    = document.getElementById("preview-image");
  const dropContent   = document.getElementById("drop-content");
  const clearBtn      = document.getElementById("clear-btn");
  const segmentBtn    = document.getElementById("segment-btn");
  const btnText       = segmentBtn.querySelector(".btn-text");
  const btnLoader     = document.getElementById("btn-loader");
  const uploadForm    = document.getElementById("upload-form");
  const uploadSection = document.getElementById("upload-section");
  const progressSec   = document.getElementById("progress-section");
  const progressBar   = document.getElementById("progress-bar");
  const progressTitle = document.getElementById("progress-title");
  const resultsSec    = document.getElementById("results-section");
  const errorSec      = document.getElementById("error-section");
  const errorMsg      = document.getElementById("error-message");
  const newScanBtn    = document.getElementById("new-scan-btn");
  const errorRetry    = document.getElementById("error-retry-btn");
  const steps         = document.querySelectorAll(".step");

  let selectedFile = null;

  // ---- File selection & preview ----
  dropZone.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", (e) => handleFile(e.target.files[0]));

  dropZone.addEventListener("dragover", (e) => { e.preventDefault(); dropZone.classList.add("drag-over"); });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
  });

  function handleFile(file) {
    if (!file || !file.type.startsWith("image/")) return;
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImg.src = e.target.result;
      previewWrap.classList.remove("hidden");
      dropContent.classList.add("hidden");
      segmentBtn.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  clearBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    resetUpload();
  });

  function resetUpload() {
    selectedFile = null;
    fileInput.value = "";
    previewWrap.classList.add("hidden");
    dropContent.classList.remove("hidden");
    segmentBtn.disabled = true;
  }

  // ---- Simulated progress ----
  let progressInterval = null;
  function startProgress() {
    progressSec.classList.remove("hidden");
    let pct = 0, step = 0;
    const stepThresholds = [15, 35, 65, 85, 95];
    progressInterval = setInterval(() => {
      pct = Math.min(pct + Math.random() * 3 + 0.5, 95);
      progressBar.style.width = pct + "%";
      const newStep = stepThresholds.findIndex((t) => pct < t);
      const activeStep = newStep === -1 ? 4 : newStep;
      steps.forEach((s, i) => {
        s.classList.remove("active", "done");
        if (i < activeStep) s.classList.add("done");
        else if (i === activeStep) s.classList.add("active");
      });
    }, 300);
  }
  function finishProgress() {
    clearInterval(progressInterval);
    progressBar.style.width = "100%";
    steps.forEach((s) => { s.classList.remove("active"); s.classList.add("done"); });
    setTimeout(() => progressSec.classList.add("hidden"), 400);
  }

  // ---- Form submission ----
  uploadForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!selectedFile) return;

    // UI state: loading
    btnText.textContent = "Processing…";
    btnLoader.classList.remove("hidden");
    segmentBtn.disabled = true;
    errorSec.classList.add("hidden");
    resultsSec.classList.add("hidden");
    startProgress();

    const formData = new FormData();
    formData.append("image", selectedFile);
    formData.append("preset", document.getElementById("preset-select").value);
    formData.append("label_hint", document.getElementById("label-select").value);

    try {
      const res = await fetch("/api/segment", { method: "POST", body: formData });
      const data = await res.json();

      if (!res.ok) throw new Error(data.error || "Server error");

      finishProgress();
      showResults(data);
    } catch (err) {
      finishProgress();
      showError(err.message);
    } finally {
      btnText.textContent = "Run Segmentation";
      btnLoader.classList.add("hidden");
      segmentBtn.disabled = false;
    }
  });

  // ---- Display results ----
  function showResults(data) {
    document.getElementById("result-original").src      = "data:image/png;base64," + data.original;
    document.getElementById("result-overlay").src       = "data:image/png;base64," + data.tumor_overlay;
    document.getElementById("result-clustered").src     = "data:image/png;base64," + data.clustered;
    document.getElementById("result-mask").src          = "data:image/png;base64," + data.tumor_mask;
    document.getElementById("result-brain").src         = "data:image/png;base64," + data.brain_mask;
    document.getElementById("result-preprocessed").src  = "data:image/png;base64," + data.preprocessed;

    const s = data.stats;
    document.getElementById("stat-clusters").textContent = s.clusters;
    document.getElementById("stat-tumor").textContent    = s.tumor_pixels.toLocaleString();
    document.getElementById("stat-area").textContent     = (s.tumor_area_fraction * 100).toFixed(2) + "%";
    document.getElementById("stat-time").textContent     = s.processing_time + "s";

    resultsSec.classList.remove("hidden");
    resultsSec.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function showError(msg) {
    errorMsg.textContent = msg;
    errorSec.classList.remove("hidden");
  }

  // ---- New scan / Retry ----
  newScanBtn.addEventListener("click", () => {
    resultsSec.classList.add("hidden");
    resetUpload();
    window.scrollTo({ top: 0, behavior: "smooth" });
  });
  errorRetry.addEventListener("click", () => {
    errorSec.classList.add("hidden");
    window.scrollTo({ top: 0, behavior: "smooth" });
  });
})();

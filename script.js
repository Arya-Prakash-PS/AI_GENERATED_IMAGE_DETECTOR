/**
 * AI Image Detector — Client-Side JavaScript
 * ============================================
 * Handles drag-and-drop uploads, file preview, API calls,
 * animated gauge rendering, and UI state management.
 */

(function () {
    "use strict";

    // ── DOM Elements ──
    const uploadSection = document.getElementById("uploadSection");
    const analysisSection = document.getElementById("analysisSection");
    const uploadZone = document.getElementById("uploadZone");
    const fileInput = document.getElementById("fileInput");
    const previewImage = document.getElementById("previewImage");
    const imageMeta = document.getElementById("imageMeta");
    const resetBtn = document.getElementById("resetBtn");
    const retryBtn = document.getElementById("retryBtn");

    // States
    const loadingState = document.getElementById("loadingState");
    const resultState = document.getElementById("resultState");
    const errorState = document.getElementById("errorState");

    // Result elements
    const gauge = document.getElementById("gauge");
    const gaugeFill = document.getElementById("gaugeFill");
    const gaugeValue = document.getElementById("gaugeValue");
    const labelBadge = document.getElementById("labelBadge");
    const confidenceValue = document.getElementById("confidenceValue");
    const detailSize = document.getElementById("detailSize");
    const detailBlur = document.getElementById("detailBlur");
    const detailEnhanced = document.getElementById("detailEnhanced");
    const enhancedRow = document.getElementById("enhancedRow");
    const errorMessage = document.getElementById("errorMessage");

    // Track current file for retry
    let currentFile = null;

    // ── Background Particles ──
    function createParticles() {
        const container = document.getElementById("bgParticles");
        const count = 20;

        for (let i = 0; i < count; i++) {
            const particle = document.createElement("div");
            particle.classList.add("particle");

            const size = Math.random() * 4 + 1;
            particle.style.width = size + "px";
            particle.style.height = size + "px";
            particle.style.left = Math.random() * 100 + "%";
            particle.style.animationDuration = Math.random() * 15 + 10 + "s";
            particle.style.animationDelay = Math.random() * 10 + "s";
            particle.style.opacity = Math.random() * 0.4 + 0.1;

            container.appendChild(particle);
        }
    }

    // ── Drag & Drop ──
    function setupDragDrop() {
        const events = ["dragenter", "dragover", "dragleave", "drop"];
        events.forEach((eventName) => {
            uploadZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
            });
        });

        uploadZone.addEventListener("dragenter", () => {
            uploadZone.classList.add("drag-over");
        });

        uploadZone.addEventListener("dragover", () => {
            uploadZone.classList.add("drag-over");
        });

        uploadZone.addEventListener("dragleave", (e) => {
            // Only remove if we're leaving the upload zone entirely
            if (!uploadZone.contains(e.relatedTarget)) {
                uploadZone.classList.remove("drag-over");
            }
        });

        uploadZone.addEventListener("drop", (e) => {
            uploadZone.classList.remove("drag-over");
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFile(files[0]);
            }
        });

        // Click to browse
        uploadZone.addEventListener("click", () => {
            fileInput.click();
        });

        fileInput.addEventListener("change", () => {
            if (fileInput.files.length > 0) {
                handleFile(fileInput.files[0]);
            }
        });
    }

    // ── File Handling ──
    function handleFile(file) {
        // Validate file type
        const validTypes = [
            "image/jpeg", "image/jpg", "image/png", "image/webp",
            "image/bmp", "image/tiff", "image/gif",
        ];
        if (!file.type.startsWith("image/")) {
            showError("Please upload an image file (JPG, PNG, WebP, BMP, TIFF, GIF).");
            return;
        }

        // Validate file size (20 MB)
        if (file.size > 20 * 1024 * 1024) {
            showError("File is too large. Maximum size is 20 MB.");
            return;
        }

        currentFile = file;

        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;

            // Show image dimensions after load
            const img = new window.Image();
            img.onload = () => {
                imageMeta.innerHTML = `
                    <span class="meta-item">📐 ${img.naturalWidth} × ${img.naturalHeight}</span>
                    <span class="meta-item">📄 ${formatFileSize(file.size)}</span>
                    <span class="meta-item">🏷️ ${file.type.split("/")[1].toUpperCase()}</span>
                `;
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);

        // Switch UI to analysis view
        showAnalysisView();

        // Send to API
        analyzeImage(file);
    }

    // ── API Call ──
    async function analyzeImage(file) {
        showLoadingState();

        const formData = new FormData();
        formData.append("image", file);

        try {
            const response = await fetch("/api/predict", {
                method: "POST",
                body: formData,
            });

            const data = await response.json();

            if (data.success) {
                showResult(data);
            } else {
                showError(data.error || "Unknown error occurred.");
            }
        } catch (err) {
            console.error("Analysis failed:", err);
            if (err.name === "TypeError" && err.message.includes("fetch")) {
                showError("Cannot connect to the server. Make sure the backend is running.");
            } else {
                showError("Network error. Please check your connection and try again.");
            }
        }
    }

    // ── UI State Management ──
    function showAnalysisView() {
        uploadSection.style.display = "none";
        analysisSection.style.display = "block";
    }

    function showUploadView() {
        analysisSection.style.display = "none";
        uploadSection.style.display = "block";
        fileInput.value = "";
        currentFile = null;
    }

    function showLoadingState() {
        loadingState.style.display = "flex";
        resultState.style.display = "none";
        errorState.style.display = "none";
    }

    function showResultState() {
        loadingState.style.display = "none";
        resultState.style.display = "flex";
        errorState.style.display = "none";
    }

    function showError(message) {
        if (analysisSection.style.display === "none") {
            showAnalysisView();
        }
        loadingState.style.display = "none";
        resultState.style.display = "none";
        errorState.style.display = "flex";
        errorMessage.textContent = message;
    }

    // ── Show Result ──
    function showResult(data) {
        showResultState();

        const pct = data.ai_generated_percentage;
        const angle = (pct / 100) * 360;

        // Determine color based on percentage
        let color, badgeBg, badgeBorder, badgeColor;
        if (pct >= 75) {
            color = "var(--color-fake)";
            badgeBg = "var(--color-fake-glow)";
            badgeBorder = "rgba(239, 68, 68, 0.35)";
            badgeColor = "var(--color-fake)";
        } else if (pct >= 50) {
            color = "var(--color-uncertain)";
            badgeBg = "var(--color-uncertain-glow)";
            badgeBorder = "rgba(245, 158, 11, 0.35)";
            badgeColor = "var(--color-uncertain)";
        } else if (pct >= 25) {
            color = "var(--color-uncertain)";
            badgeBg = "var(--color-uncertain-glow)";
            badgeBorder = "rgba(245, 158, 11, 0.35)";
            badgeColor = "var(--color-uncertain)";
        } else {
            color = "var(--color-real)";
            badgeBg = "var(--color-real-glow)";
            badgeBorder = "rgba(34, 197, 94, 0.35)";
            badgeColor = "var(--color-real)";
        }

        // Animate gauge
        gauge.style.setProperty("--gauge-color", color);

        // Animate value counting up
        animateValue(gaugeValue, 0, pct, 1000);

        // Animate gauge fill with slight delay
        setTimeout(() => {
            gaugeFill.style.background = `conic-gradient(
                ${color} 0deg,
                ${color} ${angle}deg,
                rgba(255, 255, 255, 0.06) ${angle}deg,
                rgba(255, 255, 255, 0.06) 360deg
            )`;
        }, 100);

        // Badge
        labelBadge.textContent = data.label;
        labelBadge.style.setProperty("--badge-bg", badgeBg);
        labelBadge.style.setProperty("--badge-color", badgeColor);
        labelBadge.style.setProperty("--badge-border", badgeBorder);

        // Confidence
        confidenceValue.textContent = data.confidence;

        // Details
        detailSize.textContent = data.original_dimensions || "—";
        detailBlur.textContent = data.blur_score !== undefined ? data.blur_score.toFixed(1) : "—";

        if (data.image_enhanced) {
            detailEnhanced.innerHTML = '<span class="enhanced-badge">✨ Yes</span>';
            enhancedRow.style.display = "flex";
        } else {
            detailEnhanced.textContent = "No";
            enhancedRow.style.display = "flex";
        }
    }

    // ── Animate Counter ──
    function animateValue(element, start, end, duration) {
        const startTime = performance.now();
        const range = end - start;

        function tick(currentTime) {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);

            // Ease out cubic
            const eased = 1 - Math.pow(1 - progress, 3);
            const current = start + range * eased;

            element.textContent = current.toFixed(1);

            if (progress < 1) {
                requestAnimationFrame(tick);
            } else {
                element.textContent = end.toFixed(1);
            }
        }

        requestAnimationFrame(tick);
    }

    // ── Utility ──
    function formatFileSize(bytes) {
        if (bytes < 1024) return bytes + " B";
        if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " KB";
        return (bytes / 1048576).toFixed(1) + " MB";
    }

    // ── Event Listeners ──
    resetBtn.addEventListener("click", showUploadView);

    retryBtn.addEventListener("click", () => {
        if (currentFile) {
            analyzeImage(currentFile);
        } else {
            showUploadView();
        }
    });

    // ── Initialize ──
    createParticles();
    setupDragDrop();

})();

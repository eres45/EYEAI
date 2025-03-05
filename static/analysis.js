document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const dropZone = document.getElementById('dropZone');
    const imageInput = document.getElementById('imageInput');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('imagePreview');
    const analyzeButton = document.getElementById('analyzeButton');
    const removeImage = document.getElementById('removeImage');
    const uploadForm = document.getElementById('uploadForm');
    const fileName = document.getElementById('fileName');
    const fileSize = document.getElementById('fileSize');
    const cameraOption = document.querySelector('.option-card:first-child');
    const browseOption = document.querySelector('.option-card:last-child');
    const cameraModal = document.getElementById('cameraModal');
    const cameraFeed = document.getElementById('cameraFeed');
    const captureButton = document.getElementById('captureButton');
    const closeButton = document.querySelector('.close-button');
    const loadingSection = document.getElementById('loadingSection');
    const resultsSection = document.getElementById('resultsSection');
    const progressSteps = document.querySelectorAll('.progress-steps .step');
    const themeToggle = document.getElementById('theme-toggle');
    const reanalyzeButton = document.getElementById('reanalyzeButton');
    const downloadReport = document.getElementById('downloadReport');

    // Remove theme toggle event listener
    if (themeToggle) {
        themeToggle.remove();
    }

    // Drag and Drop functionality
    if (dropZone) {
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('drag-over');
        });

        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('drag-over');
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('drag-over');
            const file = e.dataTransfer.files[0];
            if (isValidImage(file)) {
                handleImageUpload(file);
            }
        });

        dropZone.addEventListener('click', () => {
            if (imageInput) {
                imageInput.click();
            }
        });
    }

    if (browseOption) {
        browseOption.addEventListener('click', () => {
            if (imageInput) {
                imageInput.click();
            }
        });
    }

    if (imageInput) {
        imageInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (isValidImage(file)) {
                handleImageUpload(file);
            }
        });
    }

    // Camera functionality
    let stream = null;

    if (cameraOption) {
        cameraOption.addEventListener('click', async () => {
            try {
                stream = await navigator.mediaDevices.getUserMedia({ video: true });
                if (cameraFeed) {
                    cameraFeed.srcObject = stream;
                }
                if (cameraModal) {
                    cameraModal.classList.add('active');
                }
            } catch (err) {
                console.error('Error accessing camera:', err);
                alert('Unable to access camera. Please ensure camera permissions are granted.');
            }
        });
    }

    if (closeButton) {
        closeButton.addEventListener('click', () => {
            stopCamera();
            if (cameraModal) {
                cameraModal.classList.remove('active');
            }
        });
    }

    if (captureButton) {
        captureButton.addEventListener('click', () => {
            const canvas = document.createElement('canvas');
            canvas.width = cameraFeed.videoWidth;
            canvas.height = cameraFeed.videoHeight;
            canvas.getContext('2d').drawImage(cameraFeed, 0, 0);
            canvas.toBlob((blob) => {
                const file = new File([blob], 'camera-capture.jpg', { type: 'image/jpeg' });
                handleImageUpload(file);
            }, 'image/jpeg');
            stopCamera();
            if (cameraModal) {
                cameraModal.classList.remove('active');
            }
        });
    }

    // Remove image
    if (removeImage) {
        removeImage.addEventListener('click', () => {
            if (previewContainer) {
                previewContainer.classList.add('hidden');
            }
            if (analyzeButton) {
                analyzeButton.disabled = true;
            }
            if (imageInput) {
                imageInput.value = '';
            }
            if (dropZone) {
                dropZone.classList.remove('hidden');
            }
        });
    }

    // Form submission
    if (uploadForm) {
        uploadForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            // Show loading section and hide preview
            if (loadingSection) {
                loadingSection.classList.remove('hidden');
            }
            if (previewContainer) {
                previewContainer.classList.add('hidden');
            }
            
            // Reset progress steps
            if (progressSteps) {
                progressSteps.forEach(step => step.classList.remove('active'));
            }
            
            // Start progress animation
            let currentStep = 0;
            const progressInterval = setInterval(() => {
                // Reset previous step
                if (currentStep > 0) {
                    if (progressSteps) {
                        progressSteps[currentStep - 1].classList.remove('active');
                        progressSteps[currentStep - 1].classList.add('complete');
                    }
                }
                
                // Activate current step
                if (progressSteps) {
                    progressSteps[currentStep].classList.add('active');
                }
                
                // Move to next step
                currentStep++;
                
                // If we've gone through all steps, restart
                if (currentStep >= progressSteps.length) {
                    currentStep = 0;
                    if (progressSteps) {
                        progressSteps.forEach(step => {
                            step.classList.remove('active', 'complete');
                        });
                    }
                }
            }, 1000); // Change step every second
            
            // Create FormData and append the image
            const formData = new FormData();
            if (imageInput) {
                const imageFile = imageInput.files[0];
                formData.append('file', imageFile);
            }

            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();
                
                if (!response.ok) {
                    throw new Error(result.error || 'Analysis failed. Please try again.');
                }

                // Clear progress animation
                clearInterval(progressInterval);
                if (progressSteps) {
                    progressSteps.forEach(step => {
                        step.classList.remove('active');
                        step.classList.add('complete');
                    });
                }
                
                // Short delay before showing results
                await new Promise(resolve => setTimeout(resolve, 500));
                
                displayResults(result);
            } catch (error) {
                clearInterval(progressInterval);
                console.error('Error:', error);
                
                // Show error message to user
                showToast(error.message || 'An error occurred during analysis. Please try again.', 'error');
                
                // Reset UI
                if (loadingSection) {
                    loadingSection.classList.add('hidden');
                }
                if (previewContainer) {
                    previewContainer.classList.remove('hidden');
                }
                
                // Reset progress steps
                if (progressSteps) {
                    progressSteps.forEach(step => step.classList.remove('active', 'complete'));
                }
            }
        });
    }

    // Reanalyze button
    if (reanalyzeButton) {
        reanalyzeButton.addEventListener('click', () => {
            if (resultsSection) {
                resultsSection.classList.add('hidden');
            }
            if (loadingSection) {
                loadingSection.classList.add('hidden');
            }
            if (previewContainer) {
                previewContainer.classList.add('hidden');
            }
            if (dropZone) {
                dropZone.classList.remove('hidden');
            }
            
            // Reset form and preview
            if (imageInput) {
                imageInput.value = '';
            }
            if (imagePreview) {
                imagePreview.src = '';
            }
            if (fileName) {
                fileName.textContent = '';
            }
            if (fileSize) {
                fileSize.textContent = '';
            }
            
            // Reset progress steps
            if (progressSteps) {
                progressSteps.forEach(step => step.classList.remove('active'));
            }
            
            // Reset analysis results
            const elements = {
                'diagnosisText': '',
                'confidenceBar': '0%',
                'confidenceScore': '',
                'symptomsList': '',
                'recommendationBox': '',
                'visualizationImage': '',
                'analysisGraph': ''
            };
            
            for (const [id, value] of Object.entries(elements)) {
                const element = document.getElementById(id);
                if (element) {
                    if (id === 'confidenceBar') {
                        element.style.width = value;
                    } else if (id === 'symptomsList' || id === 'recommendationBox') {
                        element.innerHTML = value;
                    } else if (id === 'visualizationImage' || id === 'analysisGraph') {
                        element.src = value;
                        element.classList.add('hidden');
                    } else {
                        element.textContent = value;
                    }
                }
            }
        });
    }

    // Download report functionality
    if (downloadReport) {
        downloadReport.addEventListener('click', async () => {
            try {
                // Add loading state
                downloadReport.classList.add('loading');
                downloadReport.disabled = true;
                
                // Generate report content
                const reportContent = generateReportText();
                
                // Create blob and download
                const blob = new Blob([reportContent], { type: 'text/plain' });
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `eye_flu_report_${new Date().toISOString().split('T')[0]}.txt`;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
                
                // Show success message
                showToast('Report downloaded successfully!', 'success');
            } catch (error) {
                console.error('Error generating report:', error);
                showToast('Failed to generate report. Please try again.', 'error');
            } finally {
                // Remove loading state
                downloadReport.classList.remove('loading');
                downloadReport.disabled = false;
            }
        });
    }

    if (reanalyzeButton) {
        reanalyzeButton.addEventListener('click', () => {
            // Add loading state
            reanalyzeButton.classList.add('loading');
            
            // Reset UI elements
            resetUI();
            
            // Smooth scroll to top
            window.scrollTo({
                top: 0,
                behavior: 'smooth'
            });
            
            // Remove loading state after animation
            setTimeout(() => {
                reanalyzeButton.classList.remove('loading');
            }, 1000);
        });
    }

    function resetUI() {
        // Reset file input
        if (imageInput) {
            imageInput.value = '';
        }
        
        // Reset preview
        if (previewContainer) {
            previewContainer.classList.add('hidden');
        }
        
        // Reset results section
        if (resultsSection) {
            resultsSection.classList.add('hidden');
        }
        
        // Reset loading section
        if (loadingSection) {
            loadingSection.classList.add('hidden');
        }
        
        // Show drop zone
        if (dropZone) {
            dropZone.classList.remove('hidden');
            dropZone.classList.remove('drag-over');
        }
        
        // Reset progress steps
        if (progressSteps) {
            progressSteps.forEach(step => {
                step.classList.remove('active');
            });
        }
        
        // Reset analysis results
        const elements = {
            'diagnosisText': '',
            'confidenceBar': '0%',
            'confidenceScore': '',
            'symptomsList': '',
            'recommendationBox': '',
            'visualizationImage': '',
            'analysisGraph': ''
        };
        
        for (const [id, value] of Object.entries(elements)) {
            const element = document.getElementById(id);
            if (element) {
                if (id === 'confidenceBar') {
                    element.style.width = value;
                } else if (id === 'symptomsList' || id === 'recommendationBox') {
                    element.innerHTML = value;
                } else if (id === 'visualizationImage' || id === 'analysisGraph') {
                    element.src = value;
                    element.classList.add('hidden');
                } else {
                    element.textContent = value;
                }
            }
        }
    }

    function showToast(message, type = 'info') {
        // Create toast container if it doesn't exist
        let toastContainer = document.querySelector('.toast-container');
        if (!toastContainer) {
            toastContainer = document.createElement('div');
            toastContainer.className = 'toast-container';
            document.body.appendChild(toastContainer);
        }
        
        // Create toast element
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.textContent = message;
        
        // Add toast to container
        toastContainer.appendChild(toast);
        
        // Remove toast after animation
        setTimeout(() => {
            toast.classList.add('fade-out');
            setTimeout(() => {
                toastContainer.removeChild(toast);
                if (toastContainer.children.length === 0) {
                    document.body.removeChild(toastContainer);
                }
            }, 300);
        }, 3000);
    }

    // Add this CSS for the toast notifications
    const style = document.createElement('style');
    style.textContent = `
        .toast-container {
            position: fixed;
            bottom: 20px;
            right: 20px;
            z-index: 1000;
        }
        
        .toast {
            padding: 12px 24px;
            margin: 8px;
            border-radius: 8px;
            color: white;
            font-weight: 500;
            opacity: 1;
            transition: opacity 0.3s ease;
        }
        
        .toast.success {
            background-color: var(--success-color);
        }
        
        .toast.error {
            background-color: var(--error-color);
        }
        
        .toast.info {
            background-color: var(--primary-color);
        }
        
        .toast.fade-out {
            opacity: 0;
        }
    `;
    document.head.appendChild(style);

    // Helper functions
    function isValidImage(file) {
        const validTypes = ['image/jpeg', 'image/png', 'image/jpg'];
        if (!validTypes.includes(file.type)) {
            alert('Please upload a valid image file (JPG, PNG)');
            return false;
        }
        return true;
    }

    function handleImageUpload(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            if (imagePreview) {
                imagePreview.src = e.target.result;
            }
            if (fileName) {
                fileName.textContent = file.name;
            }
            if (fileSize) {
                fileSize.textContent = formatFileSize(file.size);
            }
            if (dropZone) {
                dropZone.classList.add('hidden');
            }
            if (previewContainer) {
                previewContainer.classList.remove('hidden');
            }
            if (analyzeButton) {
                analyzeButton.disabled = false;
            }
        };
        reader.readAsDataURL(file);
    }

    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    function stopCamera() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            stream = null;
        }
        if (cameraFeed && cameraFeed.srcObject) {
            cameraFeed.srcObject = null;
        }
    }

    async function displayResults(result) {
        try {
            if (resultsSection) {
                // Stage 1: Detection
                const detectionResult = document.getElementById('detectionResult');
                const detectionBar = document.getElementById('detectionConfidenceBar');
                if (detectionResult) {
                    detectionResult.textContent = result.has_eye_flu ? 'Eye Flu Detected' : 'No Eye Flu Detected';
                }
                if (detectionBar) {
                    detectionBar.style.width = `${result.detection_confidence}%`;
                    detectionBar.style.backgroundColor = result.has_eye_flu ? '#ff4444' : '#44ff44';
                }
                
                // Stage 2: Classification
                if (result.has_eye_flu) {
                    if (document.getElementById('diagnosisText')) {
                        document.getElementById('diagnosisText').textContent = result.condition;
                    }
                    if (document.getElementById('confidenceBar')) {
                        document.getElementById('confidenceBar').style.width = `${result.classification_confidence}%`;
                    }
                    if (document.getElementById('confidenceScore')) {
                        document.getElementById('confidenceScore').textContent = result.classification_confidence.toFixed(1);
                    }
                    
                    // Stage 3: Severity
                    if (document.getElementById('severityText')) {
                        document.getElementById('severityText').textContent = result.severity;
                    }
                    if (document.getElementById('severityConfidenceBar')) {
                        document.getElementById('severityConfidenceBar').style.width = `${result.severity_confidence}%`;
                    }
                    if (document.getElementById('severityConfidence')) {
                        document.getElementById('severityConfidence').textContent = result.severity_confidence.toFixed(1);
                    }
                    
                    // Update severity bar color based on level
                    const severityBar = document.getElementById('severityConfidenceBar');
                    if (severityBar) {
                        switch(result.severity) {
                            case 'Mild':
                                severityBar.style.backgroundColor = '#44ff44';
                                break;
                            case 'Moderate':
                                severityBar.style.backgroundColor = '#ffaa44';
                                break;
                            case 'Severe':
                                severityBar.style.backgroundColor = '#ff4444';
                                break;
                        }
                    }
                    
                    // Symptoms
                    const symptomsList = document.getElementById('symptomsList');
                    if (symptomsList) {
                        symptomsList.innerHTML = '';
                        result.symptoms.forEach(symptom => {
                            const li = document.createElement('li');
                            li.textContent = symptom;
                            symptomsList.appendChild(li);
                        });
                    }
                    
                    // Recommendations
                    const recommendationBox = document.getElementById('recommendationBox');
                    if (recommendationBox) {
                        recommendationBox.innerHTML = '';
                        result.recommendations.forEach(rec => {
                            const p = document.createElement('p');
                            p.textContent = rec;
                            recommendationBox.appendChild(p);
                        });
                    }
                    
                    // Medical References
                    const referencesList = document.getElementById('referencesList');
                    if (referencesList) {
                        referencesList.innerHTML = '';
                        if (result.medical_references) {
                            Object.entries(result.medical_references).forEach(([source, url]) => {
                                const li = document.createElement('li');
                                const a = document.createElement('a');
                                a.href = url;
                                a.target = '_blank';
                                a.textContent = `${source} Database`;
                                li.appendChild(a);
                                referencesList.appendChild(li);
                            });
                        }
                    }
                } else {
                    // Clear classification and severity sections for normal eyes
                    if (document.getElementById('diagnosisText')) {
                        document.getElementById('diagnosisText').textContent = 'Normal Eye';
                    }
                    if (document.getElementById('confidenceBar')) {
                        document.getElementById('confidenceBar').style.width = '100%';
                    }
                    if (document.getElementById('confidenceScore')) {
                        document.getElementById('confidenceScore').textContent = '100';
                    }
                    if (document.getElementById('severityText')) {
                        document.getElementById('severityText').textContent = 'N/A';
                    }
                    if (document.getElementById('severityConfidenceBar')) {
                        document.getElementById('severityConfidenceBar').style.width = '0%';
                    }
                    if (document.getElementById('severityConfidence')) {
                        document.getElementById('severityConfidence').textContent = '0';
                    }
                }
                
                // Update visualizations
                if (result.visualization) {
                    const visualizationImage = document.getElementById('visualizationImage');
                    if (visualizationImage) {
                        visualizationImage.src = `data:image/png;base64,${result.visualization}`;
                        visualizationImage.style.display = 'block';
                    }
                }

                if (result.graph) {
                    const analysisGraph = document.getElementById('analysisGraph');
                    if (analysisGraph) {
                        analysisGraph.src = `data:image/png;base64,${result.graph}`;
                        analysisGraph.style.display = 'block';
                    }
                }

                // Show results section
                if (resultsSection) {
                    resultsSection.style.display = 'block';
                    resultsSection.scrollIntoView({ behavior: 'smooth' });
                }
            }
        } catch (error) {
            console.error('Error displaying results:', error);
            showToast('Error displaying results. Please try again.', 'error');
        }
    }

    function generateReportText() {
        const date = new Date().toLocaleString();
        const detection = document.getElementById('detectionResult') ? document.getElementById('detectionResult').textContent : '';
        const diagnosis = document.getElementById('diagnosisText') ? document.getElementById('diagnosisText').textContent : '';
        const severity = document.getElementById('severityText') ? document.getElementById('severityText').textContent : '';
        const detectionConfidence = document.getElementById('detectionConfidenceBar') ? document.getElementById('detectionConfidenceBar').style.width : '';
        const classificationConfidence = document.getElementById('confidenceScore') ? document.getElementById('confidenceScore').textContent : '';
        const severityConfidence = document.getElementById('severityConfidence') ? document.getElementById('severityConfidence').textContent : '';
        
        // Get symptoms list
        const symptoms = [];
        if (document.getElementById('symptomsList')) {
            document.querySelectorAll('#symptomsList li').forEach(li => {
                symptoms.push(`- ${li.textContent}`);
            });
        }
        
        // Get recommendations
        const recommendations = [];
        if (document.getElementById('recommendationBox')) {
            document.querySelectorAll('#recommendationBox p').forEach(p => {
                recommendations.push(`- ${p.textContent}`);
            });
        }
        
        // Get medical references
        const references = [];
        if (document.getElementById('referencesList')) {
            document.querySelectorAll('#referencesList li a').forEach(a => {
                references.push(`- ${a.textContent}: ${a.href}`);
            });
        }
        
        // Create report content
        const reportContent = [
            "EYE FLU ANALYSIS REPORT",
            "=============================================",
            "",
            `Date: ${date}`,
            "",
            "STAGE 1: DETECTION",
            "---------------------------------------------",
            detection,
            `Confidence: ${detectionConfidence}`,
            "",
            "STAGE 2: CLASSIFICATION",
            "---------------------------------------------",
            diagnosis,
            `Confidence: ${classificationConfidence}%`,
            "",
            "STAGE 3: SEVERITY ASSESSMENT",
            "---------------------------------------------",
            severity,
            `Confidence: ${severityConfidence}%`,
            "",
            "SYMPTOMS",
            "---------------------------------------------",
            ...symptoms,
            "",
            "RECOMMENDATIONS",
            "---------------------------------------------",
            ...recommendations,
            "",
            "MEDICAL REFERENCES",
            "---------------------------------------------",
            ...references,
            "",
            "ADDITIONAL INFORMATION",
            "---------------------------------------------",
            "- This report was generated automatically by the Eye Flu Analysis System",
            "- Please consult a healthcare professional for medical advice",
            "- Keep this report for your records",
            "",
            "=============================================",
            "End of Report"
        ].join('\n');
        
        return reportContent;
    }
});

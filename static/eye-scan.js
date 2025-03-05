document.addEventListener('DOMContentLoaded', () => {
    // Mobile menu toggle
    const mobileToggle = document.getElementById('mobile-toggle');
    const navbarMenu = document.getElementById('navbar-menu');
    
    if (mobileToggle) {
        mobileToggle.addEventListener('click', () => {
            navbarMenu.classList.toggle('active');
            mobileToggle.classList.toggle('active');
        });
    }
    
    // DOM Elements
    const fileInput = document.getElementById('file-input');
    const uploadBtn = document.getElementById('upload-btn');
    const previewContainer = document.getElementById('preview-container');
    const previewImg = document.getElementById('preview-img');
    const analyzeBtn = document.getElementById('analyze-btn');
    const resetBtn = document.getElementById('reset-btn');
    const resultsContainer = document.getElementById('results-container');
    const comparisonOriginal = document.getElementById('comparison-original');
    const comparisonAnalyzed = document.getElementById('comparison-analyzed');
    const conditionName = document.getElementById('condition-name');
    const severityLevel = document.getElementById('severity-level');
    const confidenceScore = document.getElementById('confidence-score');
    const diagnosisDetails = document.getElementById('diagnosis-details');
    const symptomsDetected = document.getElementById('symptoms-detected');
    const analysisDate = document.getElementById('analysis-date');
    const recommendationsList = document.getElementById('recommendations-list');
    
    // Upload area elements
    const uploadArea = document.getElementById('upload-area');
    const uploadDropzone = document.getElementById('upload-dropzone');
    
    // File upload button click handler
    if (uploadBtn) {
        uploadBtn.addEventListener('click', () => {
            fileInput.click();
        });
    }
    
    // File input change handler
    if (fileInput) {
        fileInput.addEventListener('change', handleFileSelect);
    }
    
    // Drag and drop functionality
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadDropzone.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadDropzone.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        uploadDropzone.addEventListener(eventName, unhighlight, false);
    });
    
    function highlight(e) {
        uploadDropzone.classList.add('highlight');
    }
    
    function unhighlight(e) {
        uploadDropzone.classList.remove('highlight');
    }
    
    uploadDropzone.addEventListener('drop', handleDrop, false);
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }
    
    function handleFileSelect(e) {
        const files = e.target.files;
        handleFiles(files);
    }
    
    function handleFiles(files) {
        if (files.length > 0) {
            const file = files[0];
            if (file.type.startsWith('image/')) {
                const reader = new FileReader();
                
                reader.onload = function(event) {
                    previewImg.src = event.target.result;
                    previewContainer.style.display = 'block';
                    uploadArea.style.display = 'none';
                };
                
                reader.readAsDataURL(file);
            } else {
                alert('Please upload an image file.');
            }
        }
    }
    
    // Analysis functionality
    async function analyzeImage(file) {
        try {
            // Show loading state
            updateLoadingState(true);
            
            const formData = new FormData();
            formData.append('image', file);

            const response = await fetch('/api/analyze-eye', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Analysis failed');
            }

            const result = await response.json();
            console.log('Analysis result:', result);
            
            // Update UI with results
            displayResults(result);
            
            return result;
        } catch (error) {
            console.error('Error analyzing image:', error);
            displayError(error.message);
            throw error;
        } finally {
            updateLoadingState(false);
        }
    }

    function updateLoadingState(isLoading) {
        const analyzeBtn = document.getElementById('analyze-btn');
        const loadingText = 'Analyzing...';
        const defaultText = '<i class="fas fa-search"></i> Analyze Image';
        
        if (analyzeBtn) {
            if (isLoading) {
                analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> ' + loadingText;
                analyzeBtn.disabled = true;
            } else {
                analyzeBtn.innerHTML = defaultText;
                analyzeBtn.disabled = false;
            }
        }
        
        // Update result placeholders
        const elements = {
            'condition-name': isLoading ? 'Analyzing eye condition...' : '',
            'severity-level': isLoading ? 'Determining severity...' : '',
            'confidence-score': isLoading ? 'Calculating confidence...' : '',
            'diagnosis-details': isLoading ? 'Preparing diagnosis...' : '',
            'symptoms-detected': isLoading ? 'Detecting symptoms...' : '',
            'analysis-date': isLoading ? 'Analysis in progress...' : ''
        };
        
        Object.entries(elements).forEach(([id, text]) => {
            const element = document.getElementById(id);
            if (element && isLoading) {
                element.textContent = text;
            }
        });
    }

    function displayError(message) {
        const resultsContainer = document.getElementById('results-container');
        if (resultsContainer) {
            resultsContainer.style.display = 'block';
            resultsContainer.innerHTML = `
                <div class="error-message">
                    <i class="fas fa-exclamation-circle"></i>
                    <p>${message}</p>
                    <button onclick="location.reload()" class="retry-button">
                        <i class="fas fa-redo"></i> Try Again
                    </button>
                </div>
            `;
        }
    }

    function displayResults(results) {
        console.log('Displaying results:', results);
        
        const resultsContainer = document.getElementById('results-container');
        const elements = {
            conditionName: document.getElementById('condition-name'),
            severityLevel: document.getElementById('severity-level'),
            confidenceScore: document.getElementById('confidence-score'),
            diagnosisDetails: document.getElementById('diagnosis-details'),
            symptomsDetected: document.getElementById('symptoms-detected'),
            analysisDate: document.getElementById('analysis-date'),
            recommendationsList: document.getElementById('recommendations-list'),
            comparisonOriginal: document.getElementById('comparison-original'),
            comparisonAnalyzed: document.getElementById('comparison-analyzed')
        };

        if (!resultsContainer || !elements.conditionName) {
            console.error('Required DOM elements not found');
            return;
        }

        // Show the results section
        resultsContainer.style.display = 'block';

        // Update condition and status with color
        const statusColor = results.status_color || (results.has_condition ? 'red' : 'green');
        elements.conditionName.innerHTML = `
            <span class="status-badge" style="color: white; background-color: ${statusColor}; padding: 5px 10px; border-radius: 4px; display: inline-block; margin-bottom: 10px;">
                ${results.status || 'Unknown Status'}
            </span>
            <div style="margin-top: 5px;">
                ${results.condition || 'Unknown Condition'}
            </div>
        `;

        // Update severity with confidence
        if (elements.severityLevel) {
            elements.severityLevel.innerHTML = `
                <span class="severity-text">${results.severity || 'N/A'}</span>
                ${results.severity_confidence ? 
                    `<small class="confidence-text">(${(results.severity_confidence * 100).toFixed(1)}% confidence)</small>` 
                    : ''}
            `;
        }

        // Update confidence score
        if (elements.confidenceScore) {
            elements.confidenceScore.innerHTML = results.confidence ? 
                `<span class="confidence-value">${(results.confidence * 100).toFixed(1)}%</span>` : 
                'N/A';
        }

        // Update diagnosis details
        if (elements.diagnosisDetails) {
            elements.diagnosisDetails.textContent = results.severity_description || 'Analysis details not available';
        }

        // Update symptoms
        if (elements.symptomsDetected) {
            elements.symptomsDetected.textContent = results.status_detail || 'No symptoms information available';
        }

        // Update analysis date
        if (elements.analysisDate) {
            elements.analysisDate.textContent = results.analysis_date || new Date().toLocaleString();
        }

        // Update recommendations
        if (elements.recommendationsList) {
            elements.recommendationsList.innerHTML = '';
            const recommendations = results.recommendations || [];
            
            if (recommendations.length > 0) {
                recommendations.forEach(rec => {
                    const li = document.createElement('li');
                    li.textContent = rec;
                    elements.recommendationsList.appendChild(li);
                });
            } else {
                const defaultRecs = results.has_condition ? [
                    'Schedule an appointment with an eye specialist',
                    'Avoid touching or rubbing your eyes',
                    'Use prescribed eye drops as directed',
                    'Maintain good eye hygiene'
                ] : [
                    'Continue regular eye check-ups',
                    'Maintain good eye hygiene',
                    'Use protective eyewear when needed',
                    'Take regular breaks from screen time'
                ];
                
                defaultRecs.forEach(rec => {
                    const li = document.createElement('li');
                    li.textContent = rec;
                    elements.recommendationsList.appendChild(li);
                });
            }
        }

        // Update visualization if available
        if (results.visualization) {
            if (elements.comparisonOriginal) {
                elements.comparisonOriginal.src = document.getElementById('preview-img').src;
                elements.comparisonOriginal.style.display = 'block';
            }
            if (elements.comparisonAnalyzed) {
                elements.comparisonAnalyzed.src = `data:image/png;base64,${results.visualization}`;
                elements.comparisonAnalyzed.style.display = 'block';
            }
        }

        // Scroll to results
        resultsContainer.scrollIntoView({ behavior: 'smooth' });
    }

    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', async () => {
            try {
                // Show loading state
                updateLoadingState(true);
                
                // Get the image file
                const fileInput = document.getElementById('file-input');
                const file = fileInput.files[0];
                
                if (!file) {
                    throw new Error('Please select an image first');
                }
                
                // Analyze the image
                const results = await analyzeImage(file);
                
                // Display results
                displayResults(results);
            } catch (error) {
                console.error('Analysis failed:', error);
                displayError(error.message || 'Analysis failed. Please try again.');
            } finally {
                updateLoadingState(false);
            }
        });
    }
    
    // Reset functionality
    if (resetBtn) {
        resetBtn.addEventListener('click', () => {
            previewContainer.style.display = 'none';
            resultsContainer.style.display = 'none';
            uploadArea.style.display = 'block';
            fileInput.value = '';
            previewImg.src = '';
        });
    }
});

/**
 * Fake News Detection Web App - JavaScript
 * Handles form submission, AJAX requests, and interactive features
 */

// Global variables
let currentResults = null;

// DOM Content Loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
    setupEventListeners();
    loadModelInfo();
});

/**
 * Initialize the application
 */
function initializeApp() {
    console.log('Fake News Detection App initialized');
    
    // Check if models are loaded
    checkModelStatus();
    
    // Initialize tooltips if Bootstrap is available
    if (typeof bootstrap !== 'undefined') {
        var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }
}

/**
 * Setup event listeners
 */
function setupEventListeners() {
    // Form submission
    const form = document.getElementById('predictionForm');
    if (form) {
        form.addEventListener('submit', handleFormSubmit);
    }
    
    // Text area character counter
    const textArea = document.getElementById('newsText');
    if (textArea) {
        textArea.addEventListener('input', updateCharacterCount);
        textArea.addEventListener('paste', function() {
            setTimeout(updateCharacterCount, 100);
        });
    }
    
    // Model selection change
    const modelSelect = document.getElementById('modelSelect');
    if (modelSelect) {
        modelSelect.addEventListener('change', updateModelInfo);
    }
}

/**
 * Handle form submission
 */
async function handleFormSubmit(event) {
    event.preventDefault();
    
    const formData = new FormData(event.target);
    const text = formData.get('text').trim();
    const model = formData.get('model');
    
    // Validation
    if (!text) {
        showAlert('Please enter some text to analyze.', 'warning');
        return;
    }
    
    if (text.length < 10) {
        showAlert('Text is too short. Please enter at least 10 characters.', 'warning');
        return;
    }
    
    // Show loading
    showLoading();
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: text,
                model: model
            })
        });
        
        const result = await response.json();
        
        if (result.error) {
            showAlert(result.error, 'danger');
        } else {
            displayResults(result);
        }
        
    } catch (error) {
        console.error('Error:', error);
        showAlert('An error occurred while analyzing the text. Please try again.', 'danger');
    } finally {
        hideLoading();
    }
}

/**
 * Display prediction results
 */
function displayResults(result) {
    currentResults = result;
    
    // Update prediction
    const predictionElement = document.getElementById('predictionResult');
    const confidenceLevel = document.getElementById('confidenceLevel');
    
    if (result.prediction === 'Real News') {
        predictionElement.innerHTML = '<i class="fas fa-check-circle"></i> Real News';
        predictionElement.className = 'display-6 fw-bold prediction-real';
        confidenceLevel.innerHTML = `<span class="badge bg-success">${result.confidence_level} Confidence</span>`;
    } else {
        predictionElement.innerHTML = '<i class="fas fa-times-circle"></i> Fake News';
        predictionElement.className = 'display-6 fw-bold prediction-fake';
        confidenceLevel.innerHTML = `<span class="badge bg-danger">${result.confidence_level} Confidence</span>`;
    }
    
    // Update confidence scores
    const confidenceScores = document.getElementById('confidenceScores');
    confidenceScores.innerHTML = createConfidenceBars(result.confidence);
    
    // Update analysis details
    document.getElementById('modelUsed').textContent = result.model_used;
    document.getElementById('textLength').textContent = result.text_length;
    document.getElementById('wordCount').textContent = result.word_count;
    document.getElementById('confidenceLevelText').textContent = result.confidence_level;
    
    // Update model information
    const modelInfo = document.getElementById('modelInfo');
    if (result.model_info) {
        modelInfo.innerHTML = createModelInfoHTML(result.model_info);
    }
    
    // Show results section
    const resultsSection = document.getElementById('resultsSection');
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });
    
    // Add animation class
    resultsSection.classList.add('fade-in');
}

/**
 * Create confidence bars HTML
 */
function createConfidenceBars(confidence) {
    const realScore = Math.round(confidence['Real News'] * 100);
    const fakeScore = Math.round(confidence['Fake News'] * 100);
    
    return `
        <div class="confidence-bar">
            <div class="d-flex justify-content-between">
                <span class="confidence-label">Real News</span>
                <span class="confidence-score">${realScore}%</span>
            </div>
            <div class="progress">
                <div class="progress-bar bg-success" role="progressbar" 
                     style="width: ${realScore}%" aria-valuenow="${realScore}" 
                     aria-valuemin="0" aria-valuemax="100"></div>
            </div>
        </div>
        <div class="confidence-bar">
            <div class="d-flex justify-content-between">
                <span class="confidence-label">Fake News</span>
                <span class="confidence-score">${fakeScore}%</span>
            </div>
            <div class="progress">
                <div class="progress-bar bg-danger" role="progressbar" 
                     style="width: ${fakeScore}%" aria-valuenow="${fakeScore}" 
                     aria-valuemin="0" aria-valuemax="100"></div>
            </div>
        </div>
    `;
}

/**
 * Create model information HTML
 */
function createModelInfoHTML(modelInfo) {
    if (!modelInfo) return '';
    
    let html = '';
    
    if (modelInfo.description) {
        html += `<p class="small">${modelInfo.description}</p>`;
    }
    
    if (modelInfo.strengths && modelInfo.strengths.length > 0) {
        html += '<ul class="list-unstyled small">';
        modelInfo.strengths.forEach(strength => {
            html += `<li><i class="fas fa-check text-success"></i> ${strength}</li>`;
        });
        html += '</ul>';
    }
    
    if (modelInfo.accuracy) {
        html += `<p class="small"><strong>Accuracy:</strong> ${modelInfo.accuracy}</p>`;
    }
    
    return html;
}

/**
 * Show loading spinner
 */
function showLoading() {
    const spinner = document.getElementById('loadingSpinner');
    if (spinner) {
        spinner.style.display = 'block';
    }
    
    // Disable form
    const form = document.getElementById('predictionForm');
    if (form) {
        form.querySelector('button[type="submit"]').disabled = true;
    }
}

/**
 * Hide loading spinner
 */
function hideLoading() {
    const spinner = document.getElementById('loadingSpinner');
    if (spinner) {
        spinner.style.display = 'none';
    }
    
    // Enable form
    const form = document.getElementById('predictionForm');
    if (form) {
        form.querySelector('button[type="submit"]').disabled = false;
    }
}

/**
 * Show alert message
 */
function showAlert(message, type = 'info') {
    const alertContainer = document.querySelector('.container');
    if (!alertContainer) return;
    
    const alertHTML = `
        <div class="alert alert-${type} alert-dismissible fade show" role="alert">
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        </div>
    `;
    
    alertContainer.insertAdjacentHTML('afterbegin', alertHTML);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        const alert = alertContainer.querySelector('.alert');
        if (alert) {
            const bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        }
    }, 5000);
}

/**
 * Update character count
 */
function updateCharacterCount() {
    const textArea = document.getElementById('newsText');
    if (!textArea) return;
    
    const count = textArea.value.length;
    const formText = textArea.nextElementSibling;
    
    if (formText && formText.classList.contains('form-text')) {
        formText.textContent = `${count} characters. Enter at least 10 characters for analysis.`;
        
        // Change color based on length
        if (count < 10) {
            formText.className = 'form-text text-danger';
        } else if (count < 100) {
            formText.className = 'form-text text-warning';
        } else {
            formText.className = 'form-text text-success';
        }
    }
}

/**
 * Update model information when selection changes
 */
function updateModelInfo() {
    const modelSelect = document.getElementById('modelSelect');
    if (!modelSelect) return;
    
    const selectedModel = modelSelect.value;
    // You can add logic here to update model-specific information
    console.log('Selected model:', selectedModel);
}

/**
 * Check model status
 */
async function checkModelStatus() {
    try {
        const response = await fetch('/api/models');
        const data = await response.json();
        
        if (data.models && data.models.length > 0) {
            console.log('Models loaded:', data.models);
        } else {
            console.warn('No models available');
        }
    } catch (error) {
        console.error('Error checking model status:', error);
    }
}

/**
 * Load model information
 */
async function loadModelInfo() {
    try {
        const response = await fetch('/api/models');
        const data = await response.json();
        
        if (data.model_info) {
            console.log('Model info loaded:', data.model_info);
        }
    } catch (error) {
        console.error('Error loading model info:', error);
    }
}

/**
 * Load sample article
 */
function loadSample(type) {
    const textArea = document.getElementById('newsText');
    if (!textArea) return;
    
    const samples = {
        real: "Scientists at NASA have successfully launched a new Mars rover mission. The rover, named Perseverance, will search for signs of ancient life on the red planet and collect rock samples for future return to Earth. The mission represents a major milestone in space exploration and will help scientists better understand Mars' geological history. The rover is equipped with advanced instruments including cameras, spectrometers, and a drill to analyze the Martian surface.",
        
        fake: "BREAKING: Local man discovers that the government has been hiding aliens in underground bases for decades! Sources say that reptilian overlords are controlling world leaders and the media is covering it up. This shocking revelation explains everything from climate change to the stock market fluctuations. The truth is finally coming out and the people deserve to know what's really happening behind closed doors!"
    };
    
    textArea.value = samples[type] || '';
    updateCharacterCount();
    
    // Scroll to form
    const form = document.getElementById('predictionForm');
    if (form) {
        form.scrollIntoView({ behavior: 'smooth' });
    }
}

/**
 * Analyze another article
 */
function analyzeAgain() {
    const textArea = document.getElementById('newsText');
    const resultsSection = document.getElementById('resultsSection');
    
    if (textArea) {
        textArea.value = '';
        textArea.focus();
    }
    
    if (resultsSection) {
        resultsSection.style.display = 'none';
    }
    
    // Reset character count
    updateCharacterCount();
}

/**
 * Share results
 */
function shareResults() {
    if (!currentResults) return;
    
    const shareText = `Fake News Detection Results:\n\nPrediction: ${currentResults.prediction}\nConfidence: ${Math.round(Math.max(...Object.values(currentResults.confidence)) * 100)}%\nModel: ${currentResults.model_used}`;
    
    if (navigator.share) {
        navigator.share({
            title: 'Fake News Detection Results',
            text: shareText,
            url: window.location.href
        }).catch(error => {
            console.error('Error sharing:', error);
            copyToClipboard(shareText);
        });
    } else {
        copyToClipboard(shareText);
    }
}

/**
 * Copy text to clipboard
 */
function copyToClipboard(text) {
    if (navigator.clipboard) {
        navigator.clipboard.writeText(text).then(() => {
            showAlert('Results copied to clipboard!', 'success');
        }).catch(error => {
            console.error('Error copying to clipboard:', error);
            fallbackCopyToClipboard(text);
        });
    } else {
        fallbackCopyToClipboard(text);
    }
}

/**
 * Fallback copy to clipboard method
 */
function fallbackCopyToClipboard(text) {
    const textArea = document.createElement('textarea');
    textArea.value = text;
    document.body.appendChild(textArea);
    textArea.focus();
    textArea.select();
    
    try {
        document.execCommand('copy');
        showAlert('Results copied to clipboard!', 'success');
    } catch (error) {
        console.error('Fallback copy failed:', error);
        showAlert('Unable to copy to clipboard', 'warning');
    }
    
    document.body.removeChild(textArea);
}

/**
 * Smooth scroll to element
 */
function smoothScrollTo(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.scrollIntoView({ behavior: 'smooth' });
    }
}

/**
 * Format number with commas
 */
function formatNumber(num) {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
}

/**
 * Debounce function for performance
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Export functions for global access
window.loadSample = loadSample;
window.analyzeAgain = analyzeAgain;
window.shareResults = shareResults;

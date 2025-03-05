// Modern Chatbot with Black & White theme
document.addEventListener('DOMContentLoaded', function() {
    // Expose toggle function globally
    window.toggleChat = function() {
        const chatbot = document.getElementById('chatbot');
        window.chatOpen = !window.chatOpen;
        chatbot.classList.toggle('active', window.chatOpen);
        
        // Add animation effect
        if (window.chatOpen) {
            chatbot.style.animation = 'slideChatIn 0.3s forwards';
        } else {
            chatbot.style.animation = 'slideChatOut 0.3s forwards';
        }
    };
    
    window.handleKeyPress = function(event) {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            sendMessage();
        }
    };
    
    window.sendMessage = function() {
        const input = document.getElementById('chatInput');
        const message = input.value.trim();
        
        if (message) {
            addMessage(message, 'user');
            input.value = '';
            
            // Auto-resize input
            input.style.height = 'auto';
            
            // Show typing indicator
            const messagesContainer = document.getElementById('chatMessages');
            const typingIndicator = document.createElement('div');
            typingIndicator.className = 'message bot-message typing-indicator';
            typingIndicator.innerHTML = '<div class="dots"><span></span><span></span><span></span></div>';
            messagesContainer.appendChild(typingIndicator);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
            
            // Generate response based on user input and context with a slight delay to simulate processing
            setTimeout(() => {
                // Remove typing indicator
                typingIndicator.remove();
                
                // Process the message and generate a response
                const response = generateResponse(message);
                addMessage(response, 'bot');
            }, Math.random() * 1000 + 500); // Random delay between 500-1500ms for more natural feel
        }
    };
    
    window.addMessage = function(content, sender) {
        const messagesContainer = document.getElementById('chatMessages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        
        // Create message content
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        
        // Process content for URLs, lists, etc.
        const processedContent = processMessageContent(content);
        messageContent.innerHTML = processedContent;
        
        // Add timestamp
        const messageTime = document.createElement('div');
        messageTime.className = 'message-time';
        messageTime.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        
        // Add sender avatar for bot messages
        if (sender === 'bot') {
            const avatar = document.createElement('div');
            avatar.className = 'message-avatar';
            avatar.innerHTML = '<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 2C6.48 2 2 6.48 2 12C2 17.52 6.48 22 12 22C17.52 22 22 17.52 22 12C22 6.48 17.52 2 12 2ZM12 5C13.66 5 15 6.34 15 8C15 9.66 13.66 11 12 11C10.34 11 9 9.66 9 8C9 6.34 10.34 5 12 5ZM12 19.2C9.5 19.2 7.29 17.92 6 15.98C6.03 13.99 10 12.9 12 12.9C13.99 12.9 17.97 13.99 18 15.98C16.71 17.92 14.5 19.2 12 19.2Z" fill="currentColor"/></svg>';
            messageDiv.appendChild(avatar);
        }
        
        messageDiv.appendChild(messageContent);
        messageDiv.appendChild(messageTime);
        messagesContainer.appendChild(messageDiv);
        
        // Add entrance animation to new message
        messageDiv.style.animationDelay = '0ms';
        
        // Scroll to bottom
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    };
    
    // Function to process message content for better formatting
    function processMessageContent(content) {
        if (!content) return '';
        
        // Replace newlines with <br>
        let processed = content.replace(/\n/g, '<br>');
        
        // Convert bullet points for better display
        processed = processed.replace(/•\s(.*?)(?=<br>|$)/g, '<span class="bullet-point">•</span> $1<br>');
        
        // Make numbered lists look nicer
        processed = processed.replace(/(\d+)\.\s(.*?)(?=<br>|$)/g, '<span class="numbered-item">$1.</span> $2<br>');
        
        // Highlight important words with subtle emphasis
        const importantTerms = ['eye flu', 'conjunctivitis', 'symptom', 'treatment', 'prevention', 'diagnostic', 'infection'];
        importantTerms.forEach(term => {
            const regex = new RegExp(`\\b${term}\\b`, 'gi');
            processed = processed.replace(regex, match => `<span class="highlight-term">${match}</span>`);
        });
        
        return processed;
    }
    
    function generateResponse(message) {
        message = message.toLowerCase();
        
        // Context tracking
        if (!window.chatContext) {
            window.chatContext = {
                lastTopic: null,
                userHistory: [],
                conversationStage: "initial"
            };
        }
        
        // Add to user history
        window.chatContext.userHistory.push(message);
        
        // Define knowledge base
        const knowledgeBase = {
            greetings: [
                "👋 Hello! I'm your EyeAI assistant. How can I help with your eye health today?",
                "Welcome to EyeAI! I'm here to help with eye flu detection and treatment information.",
                "Hi there! I'm your AI eye health assistant. What eye health concerns can I address for you?"
            ],
            
            eyeFluInfo: {
                symptoms: "Common eye flu symptoms include: \n• Redness and irritation\n• Watery eyes\n• Swelling\n• Discharge\n• Light sensitivity\n• Blurred vision\n• Gritty feeling\n• Itching or burning sensation\n• Crusty eyelids (especially in the morning)",
                
                causes: "Eye flu (conjunctivitis) can be caused by:\n• Viral infections (most common)\n• Bacterial infections\n• Allergies\n• Environmental irritants\n• Contact lens wear\n• Chemical exposure\n• Autoimmune conditions\n\nRisk factors include:\n• Close contact with infected individuals\n• Poor hand hygiene\n• Seasonal allergies\n• Swimming in contaminated water",
                
                prevention: "Prevent eye flu by:\n• Regular hand washing with soap\n• Avoiding touching or rubbing your eyes\n• Not sharing personal items (towels, washcloths, eye makeup)\n• Maintaining good hygiene\n• Regular cleaning of contact lenses\n• Using protective eyewear\n• Keeping your environment clean\n• Avoiding known allergens\n• Getting adequate rest and maintaining good health",
                
                treatment: "Treatment options include:\n• Artificial tears for comfort\n• Cold/warm compresses\n• Antibiotic drops (for bacterial infections)\n• Antiviral medications (for viral cases)\n• Regular cleaning of the affected area\n• Antihistamine drops for allergic reactions\n• Over-the-counter pain relievers\n\nDuration and Recovery:\n• Viral: 7-14 days\n• Bacterial: 5-7 days with antibiotics\n• Allergic: Improves with allergen removal"
            },
            
            eyeAnatomy: {
                cornea: "The cornea is the clear front surface of the eye. It plays a crucial role in focusing light. It's vulnerable to infections like eye flu.",
                
                conjunctiva: "The conjunctiva is a thin, transparent membrane that covers the white part of the eye and the inner eyelids. It's the primary site affected by eye flu (conjunctivitis).",
                
                retina: "The retina is the light-sensitive layer at the back of the eye. It converts light into neural signals sent to the brain via the optic nerve.",
                
                iris: "The iris is the colored part of the eye that controls the size of the pupil, regulating the amount of light that enters the eye.",
                
                lens: "The lens is a transparent structure that, along with the cornea, refracts light to focus it on the retina.",
                
                general: "The eye is a complex organ with multiple parts, including the cornea, iris, lens, retina, optic nerve, sclera, and conjunctiva. Each plays a crucial role in vision."
            },
            
            services: {
                diagnostic: "Our AI-powered diagnostic system can detect eye flu with 99% accuracy. Upload a photo of your eye to receive an immediate analysis.",
                
                consultation: "We offer virtual consultations with qualified ophthalmologists within 24 hours of your diagnosis.",
                
                followUp: "Our system includes automated follow-up checks to monitor your recovery progress.",
                
                pricing: "Our basic diagnostic service is free. Premium services including specialist consultations start at $20.",
                
                process: "1. Upload a clear image of your eye\n2. Our AI analyzes the image in seconds\n3. Receive a detailed diagnostic report\n4. Get personalized treatment recommendations\n5. Access follow-up services as needed"
            },
            
            conditions: {
                dryEye: "Dry eye syndrome is a chronic lack of sufficient lubrication and moisture on the eye's surface. Symptoms include irritation, redness, and a gritty sensation. It can be treated with artificial tears, lifestyle changes, and in some cases, prescription medications.",
                
                glaucoma: "Glaucoma is a group of eye conditions that damage the optic nerve, usually due to abnormally high pressure in the eye. Early detection and treatment can prevent vision loss. Regular eye exams are crucial for diagnosis.",
                
                cataract: "A cataract is a clouding of the lens in the eye leading to a decrease in vision. Symptoms may include faded colors, blurry vision, and trouble with bright lights. Surgery is the most effective treatment.",
                
                amd: "Age-related macular degeneration (AMD) affects the macula, the central part of the retina. It causes central vision loss while peripheral vision remains intact. Treatment depends on the type and stage of AMD."
            }
        };
        
        // Advanced natural language processing
        const topics = {
            greeting: ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"],
            symptoms: ["symptom", "sign", "indication", "experiencing", "feel", "feeling", "notice", "noticing", "happening"],
            causes: ["cause", "reason", "why", "how come", "source", "origin", "result from", "lead to"],
            prevention: ["prevent", "avoid", "stop", "protect", "safe", "safety", "precaution", "measure", "guard against"],
            treatment: ["treat", "cure", "heal", "medicine", "medication", "drug", "remedy", "therapy", "recover", "help", "fix", "manage"],
            anatomy: ["anatomy", "structure", "part", "inside", "component", "cornea", "retina", "iris", "conjunctiva", "lens"],
            services: ["service", "offer", "provide", "available", "help", "assist", "support", "use", "utilize", "feature"],
            pricing: ["price", "cost", "fee", "charge", "payment", "pay", "expense", "affordable", "free", "cheap", "expensive"],
            process: ["process", "work", "function", "operate", "procedure", "step", "stage", "how", "way", "method"],
            diagnostic: ["diagnose", "diagnostic", "analysis", "analyze", "test", "testing", "detect", "check", "examine", "scan", "evaluate"],
            conditions: ["condition", "disease", "disorder", "problem", "issue", "ailment", "dry eye", "glaucoma", "cataract", "macular", "amd"],
            confirmation: ["yes", "yeah", "correct", "right", "true", "indeed", "absolutely", "agree", "ok", "okay", "sure"],
            negation: ["no", "nope", "not", "negative", "disagree", "wrong", "incorrect", "false"],
            gratitude: ["thank", "thanks", "appreciate", "grateful", "gratitude"],
            personal: ["my", "I", "me", "mine", "myself", "I am", "I'm", "I have", "I've", "I feel", "I notice"]
        };
        
        // Identify topics in message
        const identifiedTopics = [];
        for (const [topic, keywords] of Object.entries(topics)) {
            for (const keyword of keywords) {
                if (message.includes(keyword)) {
                    identifiedTopics.push(topic);
                    break;
                }
            }
        }
        
        // Maintain conversation context
        if (identifiedTopics.length > 0) {
            window.chatContext.lastTopic = identifiedTopics[0];
        }
        
        console.log("Identified topics:", identifiedTopics);
        console.log("Chat context:", window.chatContext);
        
        // If we detect a personal query, use more personalized response
        const isPersonalQuery = identifiedTopics.includes('personal');
        
        // Handle different scenarios based on identified topics
        if (identifiedTopics.includes('greeting')) {
            window.chatContext.conversationStage = "greeted";
            return knowledgeBase.greetings[Math.floor(Math.random() * knowledgeBase.greetings.length)];
        }
        else if (identifiedTopics.includes('symptoms')) {
            window.chatContext.lastTopic = 'symptoms';
            if (isPersonalQuery) {
                // Check if we should suggest eye analysis
                setTimeout(() => promptForEyeAnalysis("I notice you might be experiencing symptoms. I recommend uploading an image of your eye for a more accurate assessment."), 1000);
                return "I notice you might be experiencing eye symptoms. Common eye flu symptoms include redness, watery eyes, swelling, discharge, and light sensitivity. Would you like me to provide more information about these symptoms?";
            }
            return knowledgeBase.eyeFluInfo.symptoms;
        }
        else if (identifiedTopics.includes('causes')) {
            window.chatContext.lastTopic = 'causes';
            if (isPersonalQuery) {
                return "Eye flu can be caused by viral infections, bacterial infections, allergies, or environmental irritants. Based on your symptoms, a professional diagnosis would be needed to determine the exact cause. Would you like information on how our diagnostic service works?";
            }
            return knowledgeBase.eyeFluInfo.causes;
        }
        else if (identifiedTopics.includes('prevention')) {
            window.chatContext.lastTopic = 'prevention';
            return knowledgeBase.eyeFluInfo.prevention;
        }
        else if (identifiedTopics.includes('treatment')) {
            window.chatContext.lastTopic = 'treatment';
            if (isPersonalQuery) {
                return "For your eye condition, treatment would depend on the specific diagnosis. Generally, treatments include artificial tears, compresses, or medication. Our AI can help identify your condition for more targeted advice. Would you like to try our diagnostic service?";
            }
            return knowledgeBase.eyeFluInfo.treatment;
        }
        else if (identifiedTopics.includes('anatomy')) {
            window.chatContext.lastTopic = 'anatomy';
            if (message.includes("cornea")) {
                return knowledgeBase.eyeAnatomy.cornea;
            }
            else if (message.includes("conjunctiva")) {
                return knowledgeBase.eyeAnatomy.conjunctiva;
            }
            else if (message.includes("retina")) {
                return knowledgeBase.eyeAnatomy.retina;
            }
            else if (message.includes("iris")) {
                return knowledgeBase.eyeAnatomy.iris;
            }
            else if (message.includes("lens")) {
                return knowledgeBase.eyeAnatomy.lens;
            }
            else {
                return knowledgeBase.eyeAnatomy.general;
            }
        }
        else if (identifiedTopics.includes('diagnostic')) {
            window.chatContext.lastTopic = 'diagnostic';
            // Suggest eye analysis when diagnostic is mentioned
            setTimeout(() => promptForEyeAnalysis("I can provide an AI-powered diagnostic analysis of your eye. Upload an image for immediate assessment."), 800);
            return "Our AI diagnostic system can analyze eye images to detect conditions with high accuracy. Would you like to upload an eye image for analysis?";
        }
        else if (identifiedTopics.includes('services') || identifiedTopics.includes('diagnostic')) {
            window.chatContext.lastTopic = 'services';
            // Suggest eye analysis with a slight delay
            setTimeout(() => {
                if (Math.random() > 0.5) { // 50% chance to show the prompt
                    promptForEyeAnalysis("Would you like to try our eye analysis service? Upload an image of your eye for an instant AI assessment.");
                }
            }, 1500);
            return "EyeAI offers an AI-powered diagnostic system with 99% accuracy, virtual ophthalmologist consultations, and automated follow-up checks. Would you like to know more about our diagnostic process or pricing?";
        }
        else if (identifiedTopics.includes('pricing')) {
            window.chatContext.lastTopic = 'pricing';
            return knowledgeBase.services.pricing;
        }
        else if (identifiedTopics.includes('process')) {
            window.chatContext.lastTopic = 'process';
            return knowledgeBase.services.process;
        }
        else if (identifiedTopics.includes('conditions')) {
            window.chatContext.lastTopic = 'conditions';
            if (message.includes("dry eye")) {
                return knowledgeBase.conditions.dryEye;
            }
            else if (message.includes("glaucoma")) {
                return knowledgeBase.conditions.glaucoma;
            }
            else if (message.includes("cataract")) {
                return knowledgeBase.conditions.cataract;
            }
            else if (message.includes("macular") || message.includes("amd")) {
                return knowledgeBase.conditions.amd;
            }
            else {
                return "I can provide information on various eye conditions like dry eye syndrome, glaucoma, cataracts, and age-related macular degeneration. Which specific condition would you like to learn more about?";
            }
        }
        else if (identifiedTopics.includes('gratitude')) {
            return "You're welcome! I'm here to help with any eye health questions. Is there anything else you'd like to know?";
        }
        else if (window.chatContext.lastTopic) {
            // Follow up based on previous context
            switch(window.chatContext.lastTopic) {
                case 'symptoms':
                    // If discussing symptoms, occasionally suggest eye analysis
                    if (!window.chatContext.eyeAnalysisPrompted && Math.random() > 0.6) {
                        setTimeout(() => promptForEyeAnalysis(), 1000);
                    }
                    return "Would you like to know more about what might be causing these symptoms or how to treat them?";
                case 'causes':
                    return "Now that you understand what causes eye flu, would you like to learn about prevention methods or treatment options?";
                case 'treatment':
                    return "In addition to treatments, prevention is also important. Would you like to know how to prevent eye flu?";
                case 'services':
                    return "Is there a specific aspect of our services you'd like to know more about? I can tell you about our diagnostic process, pricing, or treatment recommendations.";
                default:
                    return "I'm here to help with any eye health questions. Feel free to ask about symptoms, causes, treatments, or our diagnostic services.";
            }
        }
        else {
            // If we can't identify the topic
            if (message.length < 10) {
                return "I'd be happy to help with your eye health questions. Could you please provide more details about what you'd like to know?";
            } else {
                return "I understand you're asking about eye health. To provide the most helpful information, could you specify if you're interested in eye flu symptoms, causes, treatments, or our diagnostic services?";
            }
        }
        
        // Add eye condition specific logic
        if (message.includes("red eye") || message.includes("pink eye") || 
            message.includes("irritated") || message.includes("infection") || 
            message.includes("itchy eye") || message.includes("burning") || 
            message.includes("discharge") || message.includes("my eye hurts")) {
            
            // This is a likely candidate for eye analysis
            window.chatContext.lastTopic = 'symptoms';
            setTimeout(() => promptForEyeAnalysis("Based on your description, I recommend analyzing your eye for a more accurate assessment."), 1200);
            return "It sounds like you might be experiencing symptoms of an eye condition. Common eye flu symptoms include redness, watery eyes, swelling, discharge, and light sensitivity. Would you like me to provide more information about these symptoms?";
        }
    }
    
    // Helper function for conversation intelligence
    function analyzeConversationIntent(message) {
        // This would be more sophisticated in a real AI system
        // but this simple version helps simulate more intelligent responses
        
        const intents = {
            question: ['what', 'how', 'why', 'when', 'where', 'can', 'could', '?'],
            command: ['show', 'tell', 'give', 'find', 'search', 'display', 'list'],
            statement: ['i am', 'i have', 'i feel', 'my eyes', 'it is'],
            greeting: ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']
        };
        
        // Detect primary intent
        for (const [intent, keywords] of Object.entries(intents)) {
            for (const keyword of keywords) {
                if (message.toLowerCase().includes(keyword)) {
                    return intent;
                }
            }
        }
        
        return 'unknown';
    }
    
    // Auto-resize textarea
    const chatInput = document.getElementById('chatInput');
    if (chatInput) {
        chatInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = (this.scrollHeight < 150 ? this.scrollHeight : 150) + 'px';
        });
    }
    
    // Initial welcome message with delayed appearance for realistic effect
    setTimeout(() => {
        // First clear any existing messages (important for page refreshes)
        const messagesContainer = document.getElementById('chatMessages');
        if (messagesContainer) {
            messagesContainer.innerHTML = '';
        }
        
        // Show typing indicator first
        const typingIndicator = document.createElement('div');
        typingIndicator.className = 'message bot-message typing-indicator';
        typingIndicator.innerHTML = '<div class="dots"><span></span><span></span><span></span></div>';
        messagesContainer.appendChild(typingIndicator);
        
        setTimeout(() => {
            // Remove typing indicator
            typingIndicator.remove();
            
            // Display welcome message
            const welcomeMessage = "👋 Hello! I'm your EyeAI assistant with a new modern look. How can I help with your eye health questions today?";
            addMessage(welcomeMessage, 'bot');
        }, 1200);
    }, 800);
    
    // Eye Analysis Functionality
    window.handleEyeImageUpload = function(event) {
        const file = event.target.files[0];
        if (!file) return;
        
        // Check if file is an image
        if (!file.type.startsWith('image/')) {
            addMessage("Please upload an image file for eye analysis. Supported formats include JPEG, PNG, and GIF.", 'bot');
            return;
        }
        
        // Check file size (limit to 10MB)
        if (file.size > 10 * 1024 * 1024) {
            addMessage("The image file is too large. Please upload an image smaller than 10MB.", 'bot');
            return;
        }
        
        // Display the selected image
        const reader = new FileReader();
        reader.onload = function(e) {
            const modal = document.getElementById('eyeAnalysisModal');
            const preview = document.getElementById('eyeImagePreview');
            const result = document.getElementById('eyeAnalysisResult');
            
            // Create a new image to check dimensions
            const img = new Image();
            img.onload = function() {
                // Check if image dimensions are reasonable for analysis
                if (img.width < 100 || img.height < 100) {
                    addMessage("The image resolution is too low for accurate analysis. Please upload a clearer image of your eye.", 'bot');
                    return;
                }
                
                // Set the image source
                preview.src = e.target.result;
                preview.className = 'eye-image-preview';
                
                // Reset the analysis state
                result.innerHTML = '';
                result.classList.remove('active');
                document.getElementById('analysisStatusText').textContent = "Analyzing your eye image...";
                document.querySelector('.analysis-spinner').style.display = 'flex';
                
                // Show the modal
                modal.classList.add('active');
                
                // Add a message to the chat
                addMessage("I'm analyzing your eye image now. This will take just a moment...", 'bot');
                
                // Process the image using our eye analysis models
                simulateEyeAnalysis();
            };
            
            img.src = e.target.result;
        };
        
        reader.onerror = function() {
            addMessage("There was an error reading your image file. Please try again with a different image.", 'bot');
        };
        
        reader.readAsDataURL(file);
    };
    
    window.closeEyeAnalysisModal = function() {
        const modal = document.getElementById('eyeAnalysisModal');
        modal.classList.remove('active');
    };
    
    window.simulateEyeAnalysis = function() {
        // Hide download report button initially
        const downloadReportBtn = document.getElementById("downloadReportBtn");
        if (downloadReportBtn) {
            downloadReportBtn.style.display = "none";
        }
        
        // Show processing state
        document.querySelector('.analysis-spinner').style.display = 'flex';
        document.getElementById('analysisStatusText').textContent = "Analyzing your eye image...";
        
        // Get the uploaded image
        const imageElement = document.getElementById('eyeImagePreview');
        
        // Add progress indicator
        const progressContainer = document.createElement('div');
        progressContainer.className = 'analysis-progress-container';
        progressContainer.innerHTML = `
            <div class="analysis-progress-bar">
                <div class="analysis-progress-fill"></div>
            </div>
            <div class="analysis-progress-text">0%</div>
        `;
        
        // Insert progress indicator after the spinner
        const spinnerElement = document.querySelector('.analysis-spinner');
        spinnerElement.parentNode.insertBefore(progressContainer, spinnerElement.nextSibling);
        
        // Get progress elements
        const progressFill = document.querySelector('.analysis-progress-fill');
        const progressText = document.querySelector('.analysis-progress-text');
        
        // Simulate progress over 3 seconds
        let progress = 0;
        const totalTime = 3000; // 3 seconds
        const interval = 50; // Update every 50ms
        const increment = interval / totalTime * 100;
        
        const progressInterval = setInterval(() => {
            progress += increment;
            const currentProgress = Math.min(Math.round(progress), 99); // Cap at 99% until complete
            
            // Update progress bar and text
            progressFill.style.width = `${currentProgress}%`;
            progressText.textContent = `${currentProgress}%`;
            
            // Update status text based on progress
            if (currentProgress < 30) {
                document.getElementById('analysisStatusText').textContent = "Analyzing image features...";
            } else if (currentProgress < 60) {
                document.getElementById('analysisStatusText').textContent = "Detecting eye conditions...";
            } else if (currentProgress < 90) {
                document.getElementById('analysisStatusText').textContent = "Evaluating severity...";
            }
            
            // If we've reached the end of the simulation time
            if (progress >= 100) {
                clearInterval(progressInterval);
                completeAnalysis();
            }
        }, interval);
        
        // Function to complete the analysis after the progress simulation
        function completeAnalysis() {
            // Process the image using our eye analysis models
            processEyeImage(imageElement)
                .then(analysisResult => {
                    // Update to 100%
                    progressFill.style.width = '100%';
                    progressText.textContent = '100%';
                    
                    // Hide spinner and progress
                    setTimeout(() => {
                        document.querySelector('.analysis-spinner').style.display = 'none';
                        progressContainer.style.display = 'none';
                        document.getElementById('analysisStatusText').textContent = "Analysis complete!";
                        
                        // Show results
                        const result = document.getElementById('eyeAnalysisResult');
                        result.classList.add('active');
                        
                        // Create and append analysis results
                        result.innerHTML = `
                            <div class="analysis-result-item">
                                <div class="analysis-result-title">Analysis Confidence</div>
                                <div class="analysis-result-value">
                                    ${analysisResult.confidence} 
                                    <span class="detection-status ${analysisResult.severity !== 'normal' ? 'detected' : 'not-detected'}">
                                        ${analysisResult.severity !== 'normal' ? '(Eye Flu Detected)' : '(No Eye Flu Detected)'}
                                    </span>
                                </div>
                            </div>
                            <div class="analysis-result-item">
                                <div class="analysis-result-title">Detected Condition</div>
                                <div class="analysis-condition ${analysisResult.severity}">
                                    <div class="condition-icon">
                                        ${analysisResult.severity === 'normal' ? 
                                            '<svg viewBox="0 0 24 24" fill="#00C853"><path d="M12 8C14.21 8 16 9.79 16 12C16 14.21 14.21 16 12 16C9.79 16 8 14.21 8 12C8 9.79 9.79 8 12 8ZM10 17L5 12L6.41 10.59L10 14.17L17.59 6.58L19 8L10 17Z"/></svg>' : 
                                            analysisResult.severity === 'warning' ? 
                                            '<svg viewBox="0 0 24 24" fill="#FF9800"><path d="M12 8C14.21 8 16 9.79 16 12C16 14.21 14.21 16 12 16C9.79 16 8 14.21 8 12C8 9.79 9.79 8 12 8ZM13 17H11V15H13V17ZM13 13H11V7H13V13Z"/></svg>' :
                                            '<svg viewBox="0 0 24 24" fill="#F44336"><path d="M12 2C6.48 2 2 6.48 2 12C2 17.52 6.48 22 12 22C17.52 22 22 17.52 22 12C22 6.48 17.52 2 12 2ZM13 17H11V15H13V17ZM13 13H11V7H13V13Z"/></svg>'
                                        }
                                    </div>
                                    <div class="condition-text">
                                        <div class="condition-severity">${analysisResult.name}</div>
                                        <div class="condition-description">${analysisResult.description}</div>
                                    </div>
                                </div>
                            </div>
                        `;
                        
                        // Store analysis result for later use
                        window.lastAnalysisResult = analysisResult;
                        
                        // Show download report button
                        const downloadReportBtn = document.getElementById("downloadReportBtn");
                        if (downloadReportBtn) {
                            downloadReportBtn.style.display = "inline-flex";
                            console.log("Download report button made visible after analysis");
                        }
                    }, 300); // Small delay for smooth transition
                })
                .catch(error => {
                    // Hide spinner and progress
                    document.querySelector('.analysis-spinner').style.display = 'none';
                    progressContainer.style.display = 'none';
                    document.getElementById('analysisStatusText').textContent = "Analysis failed";
                    console.error("Eye analysis error:", error);
                    
                    // Show error message
                    const result = document.getElementById('eyeAnalysisResult');
                    result.classList.add('active');
                    result.innerHTML = `
                        <div class="analysis-result-item error">
                            <div class="analysis-result-title">Error</div>
                            <div class="analysis-result-value">Could not analyze the image. Please try again with a clearer image.</div>
                        </div>
                    `;
                });
        }
    };

    // Function to process the eye image using models
    async function processEyeImage(imageElement) {
        try {
            console.log("Processing eye image");
            
            // Convert image to blob
            const canvas = document.createElement('canvas');
            const context = canvas.getContext('2d');
            canvas.width = imageElement.naturalWidth;
            canvas.height = imageElement.naturalHeight;
            context.drawImage(imageElement, 0, 0);
            
            const blob = await new Promise(resolve => {
                canvas.toBlob(resolve, 'image/jpeg', 0.9);
            });
            
            // Create form data
            const formData = new FormData();
            formData.append('file', blob, 'eye_image.jpg');
            
            // Send to server for analysis
            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Failed to analyze image');
            }
            
            // Get analysis results from server
            const analysisData = await response.json();
            console.log("Server analysis result:", JSON.stringify(analysisData, null, 2));
            
            // Map server response to our format
            return mapServerResponseToUI(analysisData);
        } catch (error) {
            console.error("Error processing eye image:", error);
            throw error;
        }
    }
    
    // Function to map server response to UI format
    function mapServerResponseToUI(serverResponse) {
        console.log("Full server response:", JSON.stringify(serverResponse, null, 2));
        
        // Check if the response indicates an eye condition
        if (!serverResponse.has_eye_flu) {
            console.log("No eye flu detected");
            return {
                name: "Healthy Eye",
                severity: "normal",
                description: "No significant abnormalities detected. Your eye appears normal and healthy.",
                confidence: Math.round(serverResponse.detection_confidence) + "%"
            };
        }
        
        // Map condition names from the classification model
        const conditionMap = {
            "Normal": {
                name: "Healthy Eye",
                severity: "normal",
                description: "No significant abnormalities detected. Your eye appears normal and healthy."
            },
            "Bacterial Conjunctivitis": {
                name: "Bacterial Conjunctivitis",
                severity: "warning",
                description: "Bacterial infection of the eye characterized by redness, discharge, and irritation. Usually treated with antibiotic eye drops."
            },
            "Viral Conjunctivitis": {
                name: "Viral Conjunctivitis",
                severity: "warning",
                description: "Viral infection of the eye characterized by redness, watery discharge, and often associated with upper respiratory infections. Highly contagious."
            },
            "Allergic Conjunctivitis": {
                name: "Allergic Conjunctivitis",
                severity: "warning",
                description: "Allergic reaction in the eye caused by exposure to allergens like pollen, dust, or pet dander. Characterized by itching, redness, and watery discharge."
            },
            "Dry Eye": {
                name: "Dry Eye Syndrome",
                severity: "warning",
                description: "Chronic lack of sufficient lubrication and moisture on the eye's surface. Symptoms include irritation, redness, and a gritty sensation."
            },
            "Glaucoma": {
                name: "Glaucoma",
                severity: "severe",
                description: "A condition that damages the optic nerve, usually due to abnormally high pressure in the eye."
            },
            "Cataract": {
                name: "Cataract",
                severity: "severe",
                description: "Clouding of the eye's lens that affects vision, typically developing slowly with age."
            },
            "Diabetic Retinopathy": {
                name: "Diabetic Retinopathy",
                severity: "severe",
                description: "Diabetes complication that affects the eyes, caused by damage to blood vessels in the retina."
            }
        };
        
        // Get the condition info from the server response
        const condition = serverResponse.condition;
        console.log("Detected condition from server:", condition);
        console.log("Available conditions in map:", Object.keys(conditionMap));
        
        // Get condition info from our map, with fallback
        let conditionInfo;
        
        if (condition && conditionMap[condition]) {
            conditionInfo = conditionMap[condition];
            console.log("Using condition from server:", condition);
        } else {
            // If condition is not found in our map, create a generic one based on the server response
            conditionInfo = {
                name: condition || "Eye Condition",
                severity: "warning",
                description: `Eye condition detected with ${Math.round(serverResponse.classification_confidence)}% confidence.`
            };
            console.log("Created generic condition info for:", condition);
        }
        
        // Adjust severity based on server's severity level
        if (serverResponse.severity === "Severe") {
            conditionInfo.severity = "severe";
        } else if (serverResponse.severity === "Mild") {
            conditionInfo.severity = "warning";
        }
        
        // Return formatted response with additional details
        return {
            name: conditionInfo.name,
            severity: conditionInfo.severity,
            description: conditionInfo.description,
            confidence: Math.round(serverResponse.classification_confidence || serverResponse.detection_confidence) + "%"
        };
    }
    
    // Function to add button to message
    function addButtonsToMessage(messageElement, serverResponse, conditionInfo) {
        if (serverResponse.has_eye_flu) {
            // Add treatment plan button
            messageElement.appendChild(
                createButton("Treatment Plan", () => showTreatmentPlan(conditionInfo.name))
            );
            
            // Add Generate PDF Report button
            messageElement.appendChild(
                createButton("Generate PDF Report", () => generatePDFReport(serverResponse))
            );
        }
    }
    
    // Function to display analysis result
    function displayAnalysisResult(serverResponse) {
        console.log("Displaying analysis result for:", serverResponse);
        
        // Store the analysis result for PDF generation
        window.lastAnalysisResult = serverResponse;
        
        // Clear previous results
        const analysisResultElement = document.getElementById("eyeAnalysisResult");
        if (analysisResultElement) {
            analysisResultElement.innerHTML = '';
        } else {
            console.error("eyeAnalysisResult element not found");
            return;
        }
        
        // Show the download report button
        const downloadReportBtn = document.getElementById("downloadReportBtn");
        if (downloadReportBtn) {
            downloadReportBtn.style.display = "inline-flex"; // Show the download button
            console.log("PDF download button made visible in displayAnalysisResult");
        } else {
            console.error("Download report button not found in the DOM");
        }
        
        // Create and add result elements
        const result = {
            name: serverResponse.condition || "Normal Eye",
            confidence: Math.round(serverResponse.classification_confidence || 98),
            severity: serverResponse.has_eye_flu ? "warning" : "normal",
            description: serverResponse.description || "Your eye appears to be in good health."
        };
        
        // Create result container
        const resultContainer = document.createElement("div");
        resultContainer.className = "analysis-result-container";
        
        // Create condition name element
        const conditionName = document.createElement("h3");
        conditionName.className = `condition-name ${result.severity}`;
        conditionName.textContent = result.name;
        resultContainer.appendChild(conditionName);
        
        // Create confidence element
        const confidence = document.createElement("div");
        confidence.className = "confidence";
        confidence.textContent = `Confidence: ${result.confidence}%`;
        resultContainer.appendChild(confidence);
        
        // Add the result container to the analysis result element
        analysisResultElement.appendChild(resultContainer);
        
        // Add description
        const description = document.createElement("p");
        description.className = "description";
        description.textContent = result.description;
        analysisResultElement.appendChild(description);
        
        // Add buttons for treatment plan
        if (serverResponse.has_eye_flu) {
            // Add treatment plan button
            const treatmentButton = createButton("Treatment Plan", () => showTreatmentPlan(result.name));
            analysisResultElement.appendChild(treatmentButton);
            
            console.log("Added treatment button");
        }
        
        // Update chat context for future reference
        window.chatContext = window.chatContext || {};
        window.chatContext.lastCondition = result.name;
        window.chatContext.lastSeverity = result.severity;
    }
    
    // Function to generate PDF report
    window.generatePDFReport = function(serverResponse) {
        console.log("Generating PDF report for:", serverResponse);
        
        // Validate input
        if (!serverResponse) {
            console.error("Cannot generate PDF: Missing server response data");
            addBotMessage("Error generating PDF report: Missing analysis data");
            return;
        }
        
        // Show loading indicator
        addBotMessage("Generating PDF report...");
        
        // Get image preview src
        const eyeImagePreview = document.getElementById("eyeImagePreview");
        const imageSrc = eyeImagePreview ? eyeImagePreview.src : "";
        
        if (!imageSrc) {
            console.warn("PDF generation: No eye image available");
        }
        
        // Get or create heatmap image
        let heatmapSrc = "";
        const heatmapCanvas = document.getElementById("heatmap-canvas");
        if (heatmapCanvas) {
            try {
                heatmapSrc = heatmapCanvas.toDataURL();
            } catch (error) {
                console.warn("Failed to get heatmap from canvas:", error);
            }
        } else if (serverResponse.heatmap_base64) {
            heatmapSrc = "data:image/png;base64," + serverResponse.heatmap_base64;
        }
        
        // Get analysis result text
        const eyeAnalysisResult = document.getElementById("eyeAnalysisResult");
        const analysisText = eyeAnalysisResult ? eyeAnalysisResult.innerText : "";
        
        // Handle cases where properties might be undefined
        const condition = serverResponse.condition || "Normal Eye";
        const confidence = serverResponse.classification_confidence 
            ? Math.round(serverResponse.classification_confidence) + "%" 
            : "98%";
        const detectionStatus = serverResponse.has_eye_flu 
            ? "Eye Flu Detected" 
            : "Healthy Eye";
        const severity = serverResponse.severity || "Normal";
        
        // Gather data for the report
        const reportData = {
            image_data: imageSrc,
            heatmap_data: heatmapSrc,
            analysis_result: analysisText || condition,
            condition: condition,
            confidence: confidence,
            detection_status: detectionStatus,
            severity: severity,
            report_type: serverResponse.has_eye_flu ? "Eye Flu Report" : "Eye Health Report",
            analysis_date: new Date().toLocaleString()
        };
        
        // Show sending message
        console.log("Sending report data for PDF generation...");
        
        // Send request to generate PDF
        fetch("/generate-pdf-report", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(reportData)
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`Failed to generate PDF report: ${response.status} ${response.statusText}`);
            }
            console.log("PDF generated successfully, getting blob response");
            return response.blob();
        })
        .then(blob => {
            if (!blob || blob.size === 0) {
                throw new Error("Received empty PDF file");
            }
            
            console.log(`Received PDF blob: ${blob.size} bytes`);
            
            // Create a download link for the PDF
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.style.display = "none";
            a.href = url;
            
            // Set appropriate filename based on report type
            const filename = serverResponse.has_eye_flu 
                ? `eye_flu_report_${new Date().toISOString().slice(0,10)}.pdf` 
                : `eye_health_report_${new Date().toISOString().slice(0,10)}.pdf`;
            a.download = filename;
            
            document.body.appendChild(a);
            a.click();
            
            // Clean up
            setTimeout(() => {
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
            }, 100);
            
            // Show success message
            addBotMessage(`${reportData.report_type} generated successfully. Download started.`);
        })
        .catch(error => {
            // Show error message
            console.error("Error generating PDF report:", error);
            addBotMessage("Error generating PDF report: " + error.message);
        });
    };

    // Function to generate a simplified PDF report (for debugging)
    window.generateSimplePDFReport = function(serverResponse) {
        console.log("Generating simplified PDF report for:", serverResponse);
        
        // Validate input
        if (!serverResponse) {
            console.error("Cannot generate simplified PDF: Missing server response data");
            addBotMessage("Error generating simplified PDF report: Missing analysis data");
            return;
        }
        
        // Show loading indicator
        addBotMessage("Generating simplified PDF report...");
        
        // Handle cases where properties might be undefined
        const condition = serverResponse.condition || "Normal Eye";
        const confidence = serverResponse.classification_confidence 
            ? Math.round(serverResponse.classification_confidence) + "%" 
            : "98%";
        const detectionStatus = serverResponse.has_eye_flu 
            ? "Eye Flu Detected" 
            : "Healthy Eye";
        
        // Get analysis result text
        const eyeAnalysisResult = document.getElementById("eyeAnalysisResult");
        const analysisText = eyeAnalysisResult ? eyeAnalysisResult.innerText : "";
        
        // Gather data for the report (no images)
        const reportData = {
            analysis_result: analysisText || condition,
            condition: condition,
            confidence: confidence,
            detection_status: detectionStatus,
            report_type: serverResponse.has_eye_flu ? "Eye Flu Report" : "Eye Health Report",
            analysis_date: new Date().toLocaleString()
        };
        
        // Show sending message
        console.log("Sending data for simplified PDF generation...");
        
        // Send request to generate PDF
        fetch("/simple-pdf-report", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(reportData)
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`Failed to generate simplified PDF report: ${response.status} ${response.statusText}`);
            }
            console.log("Simplified PDF generated successfully, getting blob response");
            return response.blob();
        })
        .then(blob => {
            if (!blob || blob.size === 0) {
                throw new Error("Received empty PDF file");
            }
            
            console.log(`Received PDF blob: ${blob.size} bytes`);
            
            // Create a download link for the PDF
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.style.display = "none";
            a.href = url;
            
            // Set appropriate filename
            const filename = `simple_eye_report_${new Date().toISOString().slice(0,10)}.pdf`;
            a.download = filename;
            
            document.body.appendChild(a);
            a.click();
            
            // Clean up
            setTimeout(() => {
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
            }, 100);
            
            // Show success message
            addBotMessage(`Simplified PDF report generated successfully. Download started.`);
        })
        .catch(error => {
            // Show error message
            console.error("Error generating simplified PDF report:", error);
            addBotMessage("Error generating simplified PDF report: " + error.message);
        });
    };

    // Add debugging button for simplified PDF on eye report click
    const originalGenerateEyeReport = window.generateEyeReport;
    window.generateEyeReport = function(serverResponse) {
        console.log("Eye report button clicked");
        
        // Add a debug message with a button to generate a simplified PDF report
        setTimeout(() => {
            const simplePdfButton = document.createElement("button");
            simplePdfButton.textContent = "Try Simple PDF";
            simplePdfButton.classList.add("bot-button");
            simplePdfButton.onclick = function() {
                window.generateSimplePDFReport(serverResponse);
            };
            
            const debugMessage = document.createElement("div");
            debugMessage.innerHTML = "Debugging: Try generating a simplified PDF report:";
            debugMessage.classList.add("bot-message");
            debugMessage.appendChild(simplePdfButton);
            
            const chatArea = document.querySelector(".chat-area");
            if (chatArea) {
                chatArea.appendChild(debugMessage);
                chatArea.scrollTop = chatArea.scrollHeight;
            }
        }, 1000);
        
        // Call the original function
        return originalGenerateEyeReport(serverResponse);
    };
    
    // Function to create a button element
    function createButton(text, onClick, className = "action-button") {
        const button = document.createElement("button");
        button.className = className;
        button.textContent = text;
        button.addEventListener("click", onClick);
        return button;
    }
    
    window.addBotMessage = function(message) {
        // Use the global addMessage if available, otherwise create a simple fallback
        if (typeof addMessage === 'function') {
            addMessage(message, 'bot');
        } else {
            // Fallback implementation if addMessage is not defined
            console.log("Bot message (fallback):", message);
            const chatMessages = document.getElementById('chatMessages');
            if (chatMessages) {
                const messageElement = document.createElement('div');
                messageElement.className = 'message bot-message';
                messageElement.innerHTML = `<div class="message-content">${message}</div>`;
                chatMessages.appendChild(messageElement);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
        }
    };
    
    // Function to prompt user for eye analysis
    function promptForEyeAnalysis(reason) {
        // Check if we've already prompted recently
        if (window.chatContext && window.chatContext.eyeAnalysisPrompted) {
            const lastPromptTime = window.chatContext.eyeAnalysisPromptTime || 0;
            const timeSincePrompt = Date.now() - lastPromptTime;
            
            // Don't prompt again if it's been less than 5 minutes
            if (timeSincePrompt < 5 * 60 * 1000) {
                return;
            }
        }
        
        // Create the prompt element
        const promptElement = document.createElement('div');
        promptElement.className = 'eye-analysis-prompt';
        promptElement.innerHTML = `
            <div class="analysis-prompt">
                <div class="prompt-icon">
                    <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M12 8C14.21 8 16 9.79 16 12C16 14.21 14.21 16 12 16C9.79 16 8 14.21 8 12C8 9.79 9.79 8 12 8ZM13 17H11V15H13V17ZM13 13H11V7H13V13Z"/>
                    </svg>
                </div>
                <div class="prompt-text">Upload an image of your eye for AI-powered analysis</div>
                <label for="eyeImageUploadPrompt" class="prompt-action">Upload Image</label>
                <input type="file" id="eyeImageUploadPrompt" accept="image/*" style="display: none;" onchange="handleEyeImageUpload(event)">
                <div class="prompt-info">
                    Our AI can detect:
                    <ul>
                        <li>Conjunctivitis</li>
                        <li>Cataracts</li>
                        <li>Glaucoma</li>
                        <li>Diabetic Retinopathy</li>
                        <li>Dry Eye Syndrome</li>
                    </ul>
                </div>
            </div>
        `;
        
        // Add the prompt to the chat
        const chatMessages = document.querySelector('.chat-messages');
        chatMessages.appendChild(promptElement);
        
        // Scroll to the bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        // Update context to prevent repeated prompts
        if (!window.chatContext) {
            window.chatContext = {};
        }
        window.chatContext.eyeAnalysisPrompted = true;
        window.chatContext.eyeAnalysisPromptTime = Date.now();
    }
    
    window.completeEyeAnalysis = function() {
        // Close the modal
        closeEyeAnalysisModal();
        
        // Get the analysis result
        const result = window.lastAnalysisResult;
        if (!result) {
            console.error("No analysis data available for treatment plan");
            addMessage("Error: No eye analysis data available. Please analyze an image first.", 'bot');
            return;
        }
        
        // Map server response fields to expected format
        const conditionData = {
            name: result.condition || "Unknown Condition",
            confidence: Math.round(result.classification_confidence) + "%",
            description: result.description || `Potential ${result.condition} detected in the eye image.`,
            severity: result.severity || (result.has_eye_flu ? "warning" : "normal")
        };
        
        // Show treatment plan
        showTreatmentPlan(conditionData);
        
        // Update context to include the analysis
        if (!window.chatContext) {
            window.chatContext = {};
        }
        window.chatContext.eyeAnalysisComplete = true;
        window.chatContext.lastCondition = conditionData.name;
        window.chatContext.lastSeverity = conditionData.severity;
    };
}); // End of DOMContentLoaded

// Function to process image and show analysis result
function analyzeImage(imageFile) {
    console.log("Starting image analysis");
    
    // Show loading message
    const loadingMessage = addMessage("Analyzing your eye image...", 'bot');
    
    // Create a FormData object to send the image file
    const formData = new FormData();
    formData.append('image', imageFile);
    
    // Disable upload button during analysis
    if (document.getElementById('upload-btn')) {
        document.getElementById('upload-btn').disabled = true;
    }
    
    // Send the image to the server for analysis
    fetch('/analyze', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        console.log("Full server response:", JSON.stringify(data, null, 2));
        
        // Save the server response in a global variable for later use
        window.lastAnalysisResult = data;
        
        // Enable upload button
        if (document.getElementById('upload-btn')) {
            document.getElementById('upload-btn').disabled = false;
        }
        
        // Show the preview image
        displayPreviewImage(imageFile);
        
        // Convert base64 data to image and display heatmap
        if (data.heatmap_base64) {
            displayHeatmap(data.heatmap_base64);
        }
        
        // Display analysis result
        displayAnalysisResult(data);
        
        // Always show the download report button, regardless of condition
        const downloadReportBtn = document.getElementById('downloadReportBtn');
        if (downloadReportBtn) {
            downloadReportBtn.style.display = 'inline-flex';
            // Make sure the button text is appropriate
            downloadReportBtn.innerHTML = `
                <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M19 9h-4V3H9v6H5l7 7 7-7zm-8 2V5h2v6h1.17L12 13.17 9.83 11H11zm-6 7h14v2H5v-2z"/>
                </svg>
                ${data.has_eye_flu ? 'Download Report' : 'Download Eye Health Report'}
            `;
            console.log("PDF download button should be visible now");
        } else {
            console.error("Download report button not found in the DOM");
        }
        
        // Add result to chat
        const resultMessage = `I've analyzed your eye image. ${data.has_eye_flu ? 
            `I've detected signs of ${data.condition} with ${Math.round(data.classification_confidence)}% confidence.` : 
            'Your eye appears normal with no signs of eye flu.'}`;
        addMessage(resultMessage, 'bot');
        
        console.log("Analysis complete:", data);
    })
    .catch(error => {
        // Enable upload button
        if (document.getElementById('upload-btn')) {
            document.getElementById('upload-btn').disabled = false;
        }
        
        // Handle any errors
        console.error('Error during image analysis:', error);
        addMessage("I'm sorry, I couldn't analyze your image. Please try again.", 'bot');
    });
}

// Function to show treatment plan based on condition
function showTreatmentPlan(conditionData) {
    console.log("Showing treatment plan for:", conditionData);
    
    let treatmentText = `Based on my analysis, I've detected: **${conditionData.name}** (${conditionData.confidence} confidence).\n\n${conditionData.description}\n\n`;
    
    // Add treatment recommendations based on the condition
    if (conditionData.severity === 'normal') {
        treatmentText += "**Recommendation**: Continue with good eye care practices:\n• Regular handwashing with soap\n• Avoid touching your eyes\n• Take regular breaks from screens\n• Use proper lighting when reading or working\n• Maintain a healthy diet rich in vitamins A, C, and E";
    } else if (conditionData.name.includes("Conjunctivitis") || conditionData.name.includes("Pink Eye")) {
        treatmentText += "**Recommendation**: \n• Use cool compresses 3-4 times daily\n• Use artificial tears to relieve discomfort\n• Avoid wearing contact lenses until symptoms resolve\n• Wash hands frequently to prevent spreading\n• Avoid sharing towels or pillows\n• If symptoms persist beyond 7 days, consult an ophthalmologist";
    } else if (conditionData.name.includes("Cataract")) {
        treatmentText += "**Recommendation**: \n• Schedule an appointment with an ophthalmologist for proper evaluation\n• Use brighter lighting for reading and other activities\n• Consider anti-glare sunglasses for outdoor activities\n• Update your eyeglass prescription if needed\n• Discuss surgical options with your doctor if vision is significantly affected";
    } else if (conditionData.name.includes("Glaucoma")) {
        treatmentText += "**Recommendation**: \n• Schedule an appointment with an ophthalmologist immediately\n• This condition requires professional medical evaluation and treatment\n• Regular eye pressure checks are essential\n• Medication may be prescribed to lower eye pressure\n• Follow up regularly with your eye care professional";
    } else if (conditionData.name.includes("Diabetic Retinopathy")) {
        treatmentText += "**Recommendation**: \n• Schedule an appointment with an ophthalmologist as soon as possible\n• If you have diabetes, ensure your blood sugar is well-controlled\n• Regular eye examinations are crucial for diabetic patients\n• Early treatment can significantly reduce the risk of vision loss\n• Maintain a healthy diet and exercise regimen";
    } else if (conditionData.name.includes("Dry Eye")) {
        treatmentText += "**Recommendation**: \n• Use artificial tears or lubricating eye drops regularly\n• Take breaks when using digital devices (follow the 20-20-20 rule)\n• Use a humidifier to add moisture to the air\n• Avoid direct air from fans, heaters, or air conditioners\n• Consider omega-3 fatty acid supplements after consulting with your doctor";
    } else if (conditionData.severity === 'warning') {
        treatmentText += "**Recommendation**: \n• Use cool compresses 3-4 times daily\n• Use artificial tears to relieve discomfort\n• Avoid wearing contact lenses until symptoms resolve\n• If symptoms persist beyond 7 days, consult an ophthalmologist";
    } else {
        treatmentText += "**Recommendation**: \n• Schedule an appointment with an ophthalmologist within 24-48 hours\n• Avoid touching or rubbing your eyes\n• Discontinue contact lens wear immediately\n• You may need specialized treatment based on your condition";
    }
    
    // Add treatment plan to chat
    addMessage(treatmentText, 'bot');
}

// Function to generate PDF report
function generateEyeReport() {
    console.log("Generate eye report button clicked");
    
    // Show the report button is clicked with console log
    console.log("PDF report button clicked");
    
    // Check if we have analysis data
    if (window.lastAnalysisResult) {
        window.generatePDFReport(window.lastAnalysisResult);
    } else {
        console.error("No analysis data available");
        addBotMessage("Error: No eye analysis data available. Please analyze an image first.");
    }
}

// Make sure these functions are accessible from the window object
window.generateEyeReport = generateEyeReport;
window.generatePDFReport = window.generatePDFReport;
window.addBotMessage = window.addBotMessage;

// Function to generate PDF report
window.generatePDFReport = function(serverResponse) {
    console.log("Generating PDF report for:", serverResponse);
    
    // Validate input
    if (!serverResponse) {
        console.error("Cannot generate PDF: Missing server response data");
        addBotMessage("Error generating PDF report: Missing analysis data");
        return;
    }
    
    // Show loading indicator
    addBotMessage("Generating PDF report...");
    
    // Get image preview src
    const eyeImagePreview = document.getElementById("eyeImagePreview");
    const imageSrc = eyeImagePreview ? eyeImagePreview.src : "";
    
    if (!imageSrc) {
        console.warn("PDF generation: No eye image available");
    }
    
    // Get or create heatmap image
    let heatmapSrc = "";
    const heatmapCanvas = document.getElementById("heatmap-canvas");
    if (heatmapCanvas) {
        try {
            heatmapSrc = heatmapCanvas.toDataURL();
        } catch (error) {
            console.warn("Failed to get heatmap from canvas:", error);
        }
    } else if (serverResponse.heatmap_base64) {
        heatmapSrc = "data:image/png;base64," + serverResponse.heatmap_base64;
    }
    
    // Get analysis result text
    const eyeAnalysisResult = document.getElementById("eyeAnalysisResult");
    const analysisText = eyeAnalysisResult ? eyeAnalysisResult.innerText : "";
    
    // Handle cases where properties might be undefined
    const condition = serverResponse.condition || "Normal Eye";
    const confidence = serverResponse.classification_confidence 
        ? Math.round(serverResponse.classification_confidence) + "%" 
        : "98%";
    const detectionStatus = serverResponse.has_eye_flu 
        ? "Eye Flu Detected" 
        : "Healthy Eye";
    const severity = serverResponse.severity || "Normal";
    
    // Gather data for the report
    const reportData = {
        image_data: imageSrc,
        heatmap_data: heatmapSrc,
        analysis_result: analysisText || condition,
        condition: condition,
        confidence: confidence,
        detection_status: detectionStatus,
        severity: severity,
        report_type: serverResponse.has_eye_flu ? "Eye Flu Report" : "Eye Health Report",
        analysis_date: new Date().toLocaleString()
    };
    
    // Show sending message
    console.log("Sending report data for PDF generation...");
    
    // Send request to generate PDF
    fetch("/generate-pdf-report", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(reportData)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`Failed to generate PDF report: ${response.status} ${response.statusText}`);
        }
        console.log("PDF generated successfully, getting blob response");
        return response.blob();
    })
    .then(blob => {
        if (!blob || blob.size === 0) {
            throw new Error("Received empty PDF file");
        }
        
        console.log(`Received PDF blob: ${blob.size} bytes`);
        
        // Create a download link for the PDF
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.style.display = "none";
        a.href = url;
        
        // Set appropriate filename based on report type
        const filename = serverResponse.has_eye_flu 
            ? `eye_flu_report_${new Date().toISOString().slice(0,10)}.pdf` 
            : `eye_health_report_${new Date().toISOString().slice(0,10)}.pdf`;
        a.download = filename;
        
        document.body.appendChild(a);
        a.click();
        
        // Clean up
        setTimeout(() => {
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
        }, 100);
        
        // Show success message
        addBotMessage(`${reportData.report_type} generated successfully. Download started.`);
    })
    .catch(error => {
        // Show error message
        console.error("Error generating PDF report:", error);
        addBotMessage("Error generating PDF report: " + error.message);
    });
    
    // Check image sources before sending
    console.log("Image source lengths:");
    console.log("Original image length:", imageSrc ? imageSrc.length : 0);
    console.log("Heatmap image length:", heatmapSrc ? heatmapSrc.length : 0);
    
    // Ensure images are in base64 format
    if (imageSrc && !imageSrc.startsWith('data:image')) {
        console.error("Original image is not in correct base64 format");
        imageSrc = null;
    }
    
    if (heatmapSrc && !heatmapSrc.startsWith('data:image')) {
        console.error("Heatmap image is not in correct base64 format");
        heatmapSrc = null;
    }
}

// Timeline and Testimonials Data
const timelineData = {
    '2025': {
        image: 'https://images.unsplash.com/photo-1551601651-2a8555f1a136?auto=format&fit=crop&q=80',
        name: 'AI-Powered Retinal Analysis',
        designation: 'Latest Technology',
        quote: 'Our advanced AI system now detects early signs of retinal diseases with unprecedented accuracy, helping prevent vision loss before it begins.'
    },
    '2024': {
        image: 'https://images.unsplash.com/photo-1579684385127-1ef15d508118?auto=format&fit=crop&q=80',
        name: 'Mobile Integration',
        designation: 'Accessibility Enhancement',
        quote: 'Launched mobile app integration allowing patients to track their eye health and receive personalized care recommendations remotely.'
    },
    '2023': {
        image: 'https://images.unsplash.com/photo-1518152006812-edab29b069ac?auto=format&fit=crop&q=80',
        name: 'Machine Learning Implementation',
        designation: 'Core Development',
        quote: 'Implemented advanced machine learning algorithms to analyze eye movement patterns and detect potential neurological conditions.'
    },
    '2022': {
        image: 'https://images.unsplash.com/photo-1581093458791-9f3c3900b7e9?auto=format&fit=crop&q=80',
        name: 'Database Integration',
        designation: 'Infrastructure',
        quote: 'Developed comprehensive patient database with secure storage and real-time analysis of eye examination results.'
    },
    '2021': {
        image: 'https://images.unsplash.com/photo-1542459590-b983117b6c89?auto=format&fit=crop&q=80',
        name: 'Initial Research',
        designation: 'Project Start',
        quote: 'Began research into AI-based eye tracking systems and their potential applications in early disease detection.'
    },
    '2020': {
        image: 'https://images.unsplash.com/photo-1576091160550-2173dba999ef?auto=format&fit=crop&q=80',
        name: 'Project Conception',
        designation: 'Planning Phase',
        quote: 'Identified the need for advanced eye tracking technology in modern healthcare and began initial project planning.'
    },
    '2019': {
        image: 'https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?auto=format&fit=crop&q=80',
        name: 'Market Research',
        designation: 'Initial Phase',
        quote: 'Conducted extensive market research to understand the gaps in current eye care technology and patient needs.'
    }
};

// Globe configuration
const globeConfig = {
    pointSize: 4,
    globeColor: "#062056",
    showAtmosphere: true,
    atmosphereColor: "#FFFFFF",
    atmosphereAltitude: 0.1,
    emissive: "#062056",
    emissiveIntensity: 0.1,
    shininess: 0.9,
    polygonColor: "rgba(255,255,255,0.7)",
    ambientLight: "#38bdf8",
    directionalLeftLight: "#ffffff",
    directionalTopLight: "#ffffff",
    pointLight: "#ffffff",
    arcTime: 1000,
    arcLength: 0.9,
    rings: 1,
    maxRings: 3,
    initialPosition: { lat: 22.3193, lng: 114.1694 },
    autoRotate: true,
    autoRotateSpeed: 0.5,
};

const colors = ["#06b6d4", "#3b82f6", "#6366f1"];

const sampleArcs = [
    {
        order: 1,
        startLat: -19.885592,
        startLng: -43.951191,
        endLat: -22.9068,
        endLng: -43.1729,
        arcAlt: 0.1,
        color: colors[Math.floor(Math.random() * colors.length)],
    },
    // Add more arcs from the sample data...
];

// Globe initialization
function initGlobe() {
    try {
        // Check dependencies
        if (typeof THREE === 'undefined') {
            throw new Error('THREE.js not loaded');
        }
        if (typeof ThreeGlobe === 'undefined') {
            throw new Error('ThreeGlobe not loaded');
        }

        // Get canvas and container
        const canvas = document.getElementById('globe-visualization');
        if (!canvas) {
            throw new Error('Globe canvas not found');
        }

        const container = canvas.parentElement;
        const width = container.clientWidth;
        const height = container.clientHeight;

        // Setup renderer
        const renderer = new THREE.WebGLRenderer({
            canvas,
            antialias: true,
            alpha: true
        });
        renderer.setSize(width, height);
        renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));

        // Setup scene
        const scene = new THREE.Scene();

        // Setup camera
        const camera = new THREE.PerspectiveCamera(50, width / height, 1, 1000);
        camera.position.z = 400;

        // Create globe
        const globe = new ThreeGlobe();
        
        // Configure globe appearance
        globe
            .globeImageUrl('https://raw.githubusercontent.com/chrisrzhou/react-globe/main/textures/globe_dark.jpg')
            .bumpImageUrl('https://raw.githubusercontent.com/chrisrzhou/react-globe/main/textures/globe_topology.png')
            .showGlobe(true)
            .showAtmosphere(true)
            .atmosphereColor('#60a5fa')
            .atmosphereAltitude(0.15)
            .globeMaterial(new THREE.MeshPhongMaterial({
                color: '#1d4ed8',
                emissive: '#1e40af',
                emissiveIntensity: 0.1,
                shininess: 0.7,
                transparent: true,
                opacity: 0.95
            }));

        // Add globe to scene
        scene.add(globe);

        // Add lights
        const ambientLight = new THREE.AmbientLight('#60a5fa', 0.6);
        scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight('#ffffff', 1);
        directionalLight.position.set(1, 1, 1);
        scene.add(directionalLight);

        const pointLight = new THREE.PointLight('#60a5fa', 1);
        pointLight.position.set(-200, 200, -200);
        scene.add(pointLight);

        // Add arcs
        const colors = ['#3b82f6', '#60a5fa', '#93c5fd'];
        const arcs = [
            {
                startLat: 20.5937,
                startLng: 78.9629,
                endLat: 40.7128,
                endLng: -74.0060,
                color: colors[0]
            },
            {
                startLat: 35.8617,
                startLng: 104.1954,
                endLat: 51.5074,
                endLng: -0.1278,
                color: colors[1]
            },
            {
                startLat: -33.8688,
                startLng: 151.2093,
                endLat: 22.3193,
                endLng: 114.1694,
                color: colors[2]
            }
        ];

        // Add points at arc endpoints
        const pointsData = [];
        arcs.forEach(arc => {
            pointsData.push(
                { lat: arc.startLat, lng: arc.startLng, size: 0.1, color: arc.color },
                { lat: arc.endLat, lng: arc.endLng, size: 0.1, color: arc.color }
            );
        });

        // Add points with glow
        globe
            .pointsData(pointsData)
            .pointColor('color')
            .pointsMerge(true)
            .pointAltitude(0.001)
            .pointRadius('size');

        // Add arcs with delay and glow effect
        setTimeout(() => {
            globe
                .arcsData(arcs)
                .arcColor('color')
                .arcDashLength(0.6)
                .arcDashGap(4)
                .arcDashInitialGap(() => Math.random() * 5)
                .arcDashAnimateTime(1500)
                .arcStroke(1)
                .arcAltitude(0.2)
                .arcAltitudeAutoScale(0.5);
        }, 1000);

        // Handle window resize
        function onWindowResize() {
            const width = container.clientWidth;
            const height = container.clientHeight;
            camera.aspect = width / height;
            camera.updateProjectionMatrix();
            renderer.setSize(width, height);
        }

        window.addEventListener('resize', onWindowResize);

        // Animation loop
        function animate() {
            renderer.render(scene, camera);
            requestAnimationFrame(animate);
        }

        // Start animation
        animate();

        console.log('Globe initialized successfully');
    } catch (error) {
        console.error('Globe initialization error:', error);
    }
}

// Timeline functionality
function initializeTimeline() {
    const yearItems = document.querySelectorAll('.year-item');
    const testimonialContainer = document.querySelector('.testimonials-container');
    const testimonialImage = document.querySelector('.testimonial-slide img');
    const testimonialName = document.querySelector('.testimonial-name');
    const testimonialDesignation = document.querySelector('.testimonial-designation');
    const testimonialQuote = document.querySelector('.testimonial-quote');
    
    if (!yearItems.length || !testimonialContainer || !testimonialImage || 
        !testimonialName || !testimonialDesignation || !testimonialQuote) {
        console.warn('Some timeline elements are missing');
        return;
    }

    let isTransitioning = false;
    let currentYear = '2025';

    function setActiveYear(year) {
        yearItems.forEach(item => {
            const isActive = item.dataset.year === year;
            item.classList.toggle('active', isActive);
            item.classList.toggle('highlight', isActive);
        });
    }

    function updateTestimonialContent(data) {
        if (!data) return;
        testimonialImage.src = data.image;
        testimonialName.textContent = data.name;
        testimonialDesignation.textContent = data.designation;
        testimonialQuote.textContent = data.quote;
    }

    function handleYearClick(year) {
        if (isTransitioning || year === currentYear) return;
        
        const data = timelineData[year];
        if (!data) return;

        isTransitioning = true;
        currentYear = year;

        setActiveYear(year);
        testimonialContainer.classList.add('transitioning');

        const tempImg = new Image();
        tempImg.onload = () => {
            setTimeout(() => {
                updateTestimonialContent(data);
                setTimeout(() => {
                    testimonialContainer.classList.remove('transitioning');
                    isTransitioning = false;
                }, 300);
            }, 150);
        };

        tempImg.onerror = () => {
            console.error('Failed to load image:', data.image);
            testimonialContainer.classList.remove('transitioning');
            isTransitioning = false;
        };

        tempImg.src = data.image;
    }

    yearItems.forEach(item => {
        item.addEventListener('click', () => {
            const year = item.dataset.year;
            if (year) {
                handleYearClick(year);
            }
        });
    });

    // Initialize with 2025
    setActiveYear('2025');
    updateTestimonialContent(timelineData['2025']);
}

// Chat functionality
let chatOpen = false;

function toggleChat() {
    const chatbot = document.getElementById('chatbot');
    chatOpen = !chatOpen;
    chatbot.classList.toggle('active', chatOpen);
}

function handleKeyPress(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

function sendMessage() {
    const input = document.getElementById('chatInput');
    const message = input.value.trim();
    
    if (message) {
        addMessage(message, 'user');
        input.value = '';
        
        // Auto-resize input
        input.style.height = 'auto';
        
        // Simulate bot response
        setTimeout(() => {
            simulateBotResponse();
        }, 1000);
    }
}

function addMessage(content, sender) {
    const messagesContainer = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    messageContent.textContent = content;
    
    const messageTime = document.createElement('div');
    messageTime.className = 'message-time';
    messageTime.textContent = 'Just now';
    
    messageDiv.appendChild(messageContent);
    messageDiv.appendChild(messageTime);
    messagesContainer.appendChild(messageDiv);
    
    // Scroll to bottom
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function simulateBotResponse() {
    const responses = [
        "I'd be happy to help you with any questions about eye health!",
        "Could you please provide more details about your concern?",
        "I can help you understand various eye conditions and treatments.",
        "Would you like to schedule an appointment with one of our specialists?",
        "I can explain our advanced AI-powered diagnostic process."
    ];
    
    const randomResponse = responses[Math.floor(Math.random() * responses.length)];
    addMessage(randomResponse, 'bot');
}

// Auto-resize textarea
document.getElementById('chatInput').addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
});

// Chatbot state and context management
let chatContext = {
    currentTopic: null,
    previousQuestions: [],
    userPreferences: {
        language: 'en',
        detailedResponses: true
    },
    conversationHistory: []
};

// Multi-language support
const languages = {
    en: 'English',
    es: 'Spanish',
    fr: 'French',
    de: 'German',
    hi: 'Hindi'
};

// Enhanced knowledge base
const knowledgeBase = {
    greetings: [
        "👋 Hello! I'm your EyeAI assistant. How can I help you with your eye health today?",
        "Welcome to EyeAI! I'm here to help you with eye flu detection and treatment information.",
        "Hi there! I'm your AI eye health assistant. Feel free to ask me anything about eye flu or our services!",
        "Hello! I'm here to help protect your eye health. What would you like to know about?"
    ],
    
    eyeFluInfo: {
        symptoms: {
            general: "Common eye flu symptoms include: \n• Redness and irritation\n• Watery eyes\n• Swelling\n• Discharge\n• Light sensitivity\n• Blurred vision\n• Gritty feeling\n• Itching or burning sensation\n• Crusty eyelids (especially in the morning)",
            
            severity: {
                mild: "Mild symptoms:\n• Slight redness\n• Minor discomfort\n• Clear discharge\n• Minimal swelling",
                moderate: "Moderate symptoms:\n• Noticeable redness\n• Consistent discomfort\n• Thick discharge\n• Visible swelling",
                severe: "Severe symptoms:\n• Intense redness\n• Severe pain\n• Heavy discharge\n• Significant swelling"
            },
            
            progression: "Symptom progression typically follows:\n1. Initial irritation and redness\n2. Development of discharge\n3. Peak symptoms (24-72 hours)\n4. Gradual improvement\n5. Resolution of symptoms"
        },
        
        causes: {
            viral: "Viral causes:\n• Adenovirus (most common)\n• Herpes simplex virus\n• Enterovirus\n• Influenza virus",
            bacterial: "Bacterial causes:\n• Staphylococcus\n• Streptococcus\n• Haemophilus\n• Pseudomonas",
            environmental: "Environmental causes:\n• Allergens\n• Air pollution\n• Chemical irritants\n• UV exposure",
            risk_factors: "Risk factors include:\n• Close contact with infected individuals\n• Poor hand hygiene\n• Seasonal allergies\n• Swimming in contaminated water\n• Contact lens use\n• Immune system weakness"
        },
        
        prevention: {
            daily_care: "Daily eye care:\n• Regular hand washing\n• Avoid touching eyes\n• Clean contact lenses properly\n• Use protective eyewear\n• Maintain good hygiene",
            
            environment: "Environmental protection:\n• Use air purifiers\n• Reduce allergen exposure\n• Maintain proper humidity\n• Clean surfaces regularly\n• Use proper lighting",
            
            lifestyle: "Lifestyle recommendations:\n• Balanced diet rich in vitamins A and C\n• Adequate sleep\n• Regular exercise\n• Stress management\n• Proper hydration",
            
            workplace: "Workplace protection:\n• Ergonomic setup\n• Regular screen breaks\n• Proper lighting\n• Protective equipment\n• Clean workstation",
            
            seasonal: "Seasonal prevention:\n• Pollen alerts monitoring\n• Indoor air quality\n• Seasonal allergies management\n• Weather protection"
        },
        
        treatment: {
            home_remedies: "Home remedies:\n• Warm/cold compresses\n• Artificial tears\n• Salt water rinse\n• Rest and hydration\n• Clean environment",
            
            medications: {
                otc: "Over-the-counter options:\n• Artificial tears\n• Antihistamine drops\n• Decongestant drops\n• Pain relievers",
                prescription: "Prescription medications:\n• Antibiotic drops/ointments\n• Antiviral medications\n• Steroid drops\n• Immunosuppressants"
            },
            
            professional: "Professional treatments:\n• Medical evaluation\n• Culture testing\n• Prescription therapy\n• Monitoring and adjustment",
            
            duration: "Treatment duration:\n• Viral: 7-14 days\n• Bacterial: 5-7 days with antibiotics\n• Allergic: Varies with exposure\n• Chemical: Depends on severity"
        },
        
        research: {
            latest: "Latest research findings:\n• New treatment approaches\n• Preventive strategies\n• Genetic factors\n• Environmental impacts",
            
            statistics: "Eye flu statistics:\n• Prevalence rates\n• Age distribution\n• Seasonal patterns\n• Treatment outcomes",
            
            studies: "Recent studies on:\n• Treatment effectiveness\n• Prevention methods\n• Risk factors\n• Long-term effects"
        }
    },
    
    services: {
        aiDiagnosis: {
            process: "AI diagnosis process:\n• Image upload\n• Symptom analysis\n• Pattern recognition\n• Severity assessment\n• Treatment recommendations",
            
            accuracy: "Diagnostic accuracy:\n• 95%+ detection rate\n• False positive < 2%\n• Continuous learning\n• Expert verification",
            
            features: "AI features:\n• Real-time analysis\n• Multi-symptom detection\n• Progress tracking\n• Predictive analytics"
        },
        
        telehealth: {
            consultations: "Virtual consultations:\n• 24/7 availability\n• Expert ophthalmologists\n• Video calls\n• Secure platform\n• Follow-up care",
            
            benefits: "Telehealth benefits:\n• Immediate access\n• Cost-effective\n• No travel needed\n• Regular monitoring\n• Specialist access",
            
            process: "Consultation process:\n1. Schedule appointment\n2. Initial assessment\n3. Video consultation\n4. Treatment plan\n5. Follow-up care"
        },
        
        monitoring: {
            tools: "Monitoring tools:\n• Symptom tracker\n• Progress charts\n• Treatment adherence\n• Recovery timeline\n• Alert system",
            
            features: "Monitoring features:\n• Daily check-ins\n• Symptom logging\n• Photo documentation\n• Treatment reminders\n• Progress reports"
        }
    },
    
    education: {
        prevention: {
            basics: "Basic prevention:\n• Hand hygiene\n• Eye protection\n• Environmental awareness\n• Regular check-ups",
            advanced: "Advanced prevention:\n• Immune system support\n• Lifestyle modifications\n• Risk management\n• Early intervention"
        },
        
        eyeHealth: {
            nutrition: "Eye health nutrition:\n• Essential vitamins\n• Minerals\n• Antioxidants\n• Dietary recommendations",
            exercise: "Eye exercises:\n• Focus training\n• Muscle strengthening\n• Strain reduction\n• Relaxation techniques"
        },
        
        resources: {
            articles: "Educational articles on:\n• Eye conditions\n• Treatment options\n• Prevention methods\n• Latest research",
            videos: "Video resources:\n• Treatment guides\n• Prevention tips\n• Expert interviews\n• Patient experiences"
        }
    }
};

// Enhanced response generation with context awareness
function generateResponse(userMessage) {
    const message = userMessage.toLowerCase();
    chatContext.previousQuestions.push(message);
    
    // Update conversation context
    updateContext(message);
    
    // Emergency check
    if (isEmergency(message)) {
        return getEmergencyResponse();
    }
    
    // Generate response based on context and message
    const response = getContextualResponse(message);
    
    // Store in conversation history
    chatContext.conversationHistory.push({
        user: userMessage,
        bot: response,
        timestamp: new Date()
    });
    
    return response;
}

function updateContext(message) {
    // Update current topic based on message content
    if (message.includes('symptom')) chatContext.currentTopic = 'symptoms';
    else if (message.includes('treat')) chatContext.currentTopic = 'treatment';
    else if (message.includes('prevent')) chatContext.currentTopic = 'prevention';
    // Add more topic detection
}

function isEmergency(message) {
    const emergencyKeywords = ['severe', 'emergency', 'urgent', 'pain', 'vision loss', 'chemical'];
    return emergencyKeywords.some(keyword => message.includes(keyword));
}

function getEmergencyResponse() {
    return "⚠️ URGENT: Based on your description, you may need immediate medical attention. Please:\n\n" +
           "1. Contact emergency services\n" +
           "2. Call our 24/7 emergency line: [Emergency Contact]\n" +
           "3. Do not delay seeking professional help\n\n" +
           "While waiting for help:\n" +
           "• Avoid rubbing your eyes\n" +
           "• Do not apply any medication without professional advice\n" +
           "• Document your symptoms\n\n" +
           "Would you like me to provide first aid instructions or connect you with our emergency team?";
}

function getContextualResponse(message) {
    // Check previous questions for context
    const hasPreviousContext = chatContext.previousQuestions.length > 1;
    const previousTopic = chatContext.currentTopic;
    
    // Generate response based on context and current query
    let response = '';
    
    if (message.match(/^(hi|hello|hey|greetings)/)) {
        return randomChoice(knowledgeBase.greetings);
    }
    
    // Symptoms with severity levels
    if (message.includes('symptom')) {
        if (message.includes('mild')) {
            response = knowledgeBase.eyeFluInfo.symptoms.severity.mild;
        } else if (message.includes('severe')) {
            response = knowledgeBase.eyeFluInfo.symptoms.severity.severe;
        } else {
            response = knowledgeBase.eyeFluInfo.symptoms.general;
        }
        
        // Add progression info if it's a follow-up question
        if (hasPreviousContext) {
            response += "\n\nTypical progression:\n" + knowledgeBase.eyeFluInfo.symptoms.progression;
        }
    }
    
    // Detailed treatment information
    if (message.includes('treat') || message.includes('medicine')) {
        if (message.includes('home') || message.includes('natural')) {
            response = knowledgeBase.eyeFluInfo.treatment.home_remedies;
        } else if (message.includes('medicine') || message.includes('drug')) {
            response = knowledgeBase.eyeFluInfo.treatment.medications.otc + "\n\n" + knowledgeBase.eyeFluInfo.treatment.medications.prescription;
        } else {
            response = knowledgeBase.eyeFluInfo.treatment.professional;
        }
        
        // Add duration info for context
        response += "\n\nTypical duration:\n" + knowledgeBase.eyeFluInfo.treatment.duration;
    }
    
    // Research and studies
    if (message.includes('research') || message.includes('study') || message.includes('new')) {
        response = knowledgeBase.eyeFluInfo.research.latest + "\n\n" + knowledgeBase.eyeFluInfo.research.statistics;
    }
    
    // Educational resources
    if (message.includes('learn') || message.includes('education') || message.includes('resource')) {
        response = knowledgeBase.education.resources.articles + "\n\n" + knowledgeBase.education.resources.videos;
    }
    
    // If no specific match, provide a contextual default response
    if (!response) {
        response = getDefaultResponse(previousTopic);
    }
    
    return response;
}

function getDefaultResponse(previousTopic) {
    if (previousTopic) {
        return `I notice we were discussing ${previousTopic}. I can provide more specific information about:\n\n` +
               `• Different aspects of ${previousTopic}\n` +
               `• Related treatments and prevention\n` +
               `• Latest research and findings\n\n` +
               `What would you like to know more about?`;
    }
    
    return "I can help you with:\n" +
           "• Detailed symptom information\n" +
           "• Treatment options and medications\n" +
           "• Prevention strategies\n" +
           "• Latest research and studies\n" +
           "• Educational resources\n\n" +
           "What specific information are you looking for?";
}

// Rest of the chat functionality (toggleChat, handleKeyPress, etc.) remains the same...

// Scroll effects
function initializeScrollEffects() {
    const fadeElements = document.querySelectorAll('.fade-up');
    if (!fadeElements.length) return;

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
            }
        });
    }, {
        threshold: 0.1
    });

    fadeElements.forEach(element => observer.observe(element));
}

// Smooth scroll functionality
function initializeSmoothScroll() {
    const links = document.querySelectorAll('a[href^="#"]');
    if (!links.length) return;

    links.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetId = link.getAttribute('href');
            if (targetId === '#') return;

            const targetElement = document.querySelector(targetId);
            if (targetElement) {
                targetElement.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

// Animations
function initializeAnimations() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('active');
                
                if (entry.target.classList.contains('process-spiral')) {
                    resetSpiralAnimation(entry.target);
                }
            } else {
                if (entry.target.classList.contains('process-spiral')) {
                    entry.target.classList.remove('active');
                }
            }
        });
    }, {
        threshold: 0.3
    });

    document.querySelectorAll('.process-spiral').forEach(spiral => {
        observer.observe(spiral);
    });
}

function resetSpiralAnimation(spiral) {
    spiral.querySelectorAll('.spiral-path').forEach(path => {
        path.style.animation = 'none';
        path.offsetHeight; 
        path.style.animation = null;
    });

    spiral.querySelectorAll('.spiral-point').forEach(point => {
        point.style.transition = 'none';
        point.offsetHeight; 
        point.style.transition = null;
    });
}

// Stat Animations
function initializeStatAnimations() {
    const stats = document.querySelectorAll('.stat-item');
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                animateStat(entry.target);
            } else {
                const numberElement = entry.target.querySelector('.stat-number');
                if (numberElement) {
                    numberElement.textContent = '0' + getStatSuffix(numberElement.textContent);
                }
            }
        });
    }, {
        threshold: 0.5
    });

    stats.forEach(stat => observer.observe(stat));
}

function getStatSuffix(text) {
    if (text.includes('%')) return '%';
    if (text.includes('s')) return 's';
    if (text.includes('k+')) return 'k+';
    return '';
}

function animateStat(stat) {
    const numberElement = stat.querySelector('.stat-number');
    if (!numberElement) return; 
    
    const finalValue = parseInt(stat.dataset.value);
    if (isNaN(finalValue)) return; 
    
    const statLabel = stat.querySelector('.stat-label');
    if (statLabel && statLabel.textContent.includes('Availability')) {
        numberElement.textContent = '0/7';
        setTimeout(() => {
            numberElement.textContent = '24/7';
        }, 2000);
        return;
    }

    let currentValue = 0;
    const duration = 2000; 
    const steps = 60;
    const increment = finalValue / steps;
    const stepDuration = duration / steps;
    const suffix = getStatSuffix(numberElement.textContent);

    if (stat.currentAnimation) {
        clearInterval(stat.currentAnimation);
    }

    stat.currentAnimation = setInterval(() => {
        currentValue += increment;
        if (currentValue >= finalValue) {
            currentValue = finalValue;
            clearInterval(stat.currentAnimation);
            stat.currentAnimation = null;
        }
        
        numberElement.textContent = Math.floor(currentValue) + suffix;
    }, stepDuration);
}

// Sliding Header
function initializeSlidingHeader() {
    const slidingHeader = document.querySelector('.sliding-header');
    const techSection = document.querySelector('.technology');
    let startY = 0;
    let currentTranslate = 0;
    const maxTranslate = 300; 

    function handleScroll() {
        if (!techSection || !slidingHeader) return;

        const rect = techSection.getBoundingClientRect();
        
        if (rect.top < window.innerHeight && rect.bottom > 0) {
            const scrollProgress = (window.innerHeight - rect.top) / (window.innerHeight + rect.height);
            
            const targetTranslate = (scrollProgress - 0.5) * maxTranslate * 2;
            
            slidingHeader.style.transform = `translateX(${targetTranslate}px)`;
        }
    }

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

    window.addEventListener('scroll', debounce(handleScroll, 10), { passive: true });
    
    handleScroll();
}

// Initialize testimonials
function initializeTestimonials(container) {
    if (!container) return;
    
    const testimonials = container.querySelectorAll('.testimonial');
    if (!testimonials.length) return;

    let currentIndex = 0;

    function showTestimonial(index) {
        testimonials.forEach((t, i) => {
            t.style.display = i === index ? 'block' : 'none';
        });
    }

    function nextTestimonial() {
        currentIndex = (currentIndex + 1) % testimonials.length;
        showTestimonial(currentIndex);
    }

    function prevTestimonial() {
        currentIndex = (currentIndex - 1 + testimonials.length) % testimonials.length;
        showTestimonial(currentIndex);
    }

    // Show first testimonial
    showTestimonial(0);

    // Add navigation if there's more than one testimonial
    if (testimonials.length > 1) {
        const nav = document.createElement('div');
        nav.className = 'testimonial-nav';
        
        const prevBtn = document.createElement('button');
        prevBtn.innerHTML = '&larr;';
        prevBtn.onclick = prevTestimonial;
        
        const nextBtn = document.createElement('button');
        nextBtn.innerHTML = '&rarr;';
        nextBtn.onclick = nextTestimonial;
        
        nav.appendChild(prevBtn);
        nav.appendChild(nextBtn);
        container.appendChild(nav);
    }
}

// CTA Button
function initializeCtaButton() {
    const ctaButton = document.querySelector('.cta-button') || document.querySelector('.minimal-cta');
    if (ctaButton) {
        ctaButton.addEventListener('click', () => {
            const fileInput = document.createElement('input');
            fileInput.type = 'file';
            fileInput.accept = 'image/*';
            
            fileInput.addEventListener('change', (e) => {
                const file = e.target.files[0];
                if (file) {
                    console.log('File selected:', file.name);
                }
            });
            
            fileInput.click();
        });
    }
}

// Safe initialization of components
function safeInitialize(selector, callback) {
    const element = document.querySelector(selector);
    if (element) {
        try {
            callback(element);
        } catch (error) {
            console.error(`Error initializing ${selector}:`, error);
        }
    }
}

// Initialize everything when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    try {
        // Initialize globe with retry
        function initGlobeWithRetry(retries = 0) {
            if (retries > 10) {
                console.error('Failed to initialize globe after multiple attempts');
                return;
            }

            if (typeof THREE === 'undefined' || typeof ThreeGlobe === 'undefined') {
                console.log('Waiting for dependencies...', { retries });
                setTimeout(() => initGlobeWithRetry(retries + 1), 500);
                return;
            }

            console.log('Dependencies loaded, initializing globe...');
            initGlobe();
        }

        // Start initialization
        initGlobeWithRetry();

        // Initialize other components
        initializeScrollEffects();
        initializeSmoothScroll();
        initializeAnimations();
        initializeStatAnimations();
        initializeSlidingHeader();
        safeInitialize('.timeline', initializeTimeline);
        safeInitialize('.chat-widget', initializeChat);
        safeInitialize('.testimonials-container', initializeTestimonials);
        safeInitialize('.cta-button', initializeCtaButton);
    } catch (error) {
        console.error('Initialization error:', error);
    }
});

// Chatbot knowledge base
const knowledgeBase = {
    greetings: [
        "👋 Hello! I'm your EyeAI assistant. How can I help you with your eye health today?",
        "Welcome to EyeAI! I'm here to help you with eye flu detection and treatment information.",
        "Hi there! I'm your AI eye health assistant. Feel free to ask me anything about eye flu or our services!",
        "Hello! I'm here to help protect your eye health. What would you like to know about?"
    ],
    
    eyeFluInfo: {
        symptoms: "Common eye flu symptoms include: \n• Redness and irritation\n• Watery eyes\n• Swelling\n• Discharge\n• Light sensitivity\n• Blurred vision\n• Gritty feeling\n• Itching or burning sensation\n• Crusty eyelids (especially in the morning)",
        
        causes: "Eye flu (conjunctivitis) can be caused by:\n• Viral infections (most common)\n• Bacterial infections\n• Allergies\n• Environmental irritants\n• Contact lens wear\n• Chemical exposure\n• Autoimmune conditions\n\nRisk factors include:\n• Close contact with infected individuals\n• Poor hand hygiene\n• Seasonal allergies\n• Swimming in contaminated water",
        
        prevention: "Prevent eye flu by:\n• Regular hand washing with soap\n• Avoiding touching or rubbing your eyes\n• Not sharing personal items (towels, washcloths, eye makeup)\n• Maintaining good hygiene\n• Regular cleaning of contact lenses\n• Using protective eyewear\n• Keeping your environment clean\n• Avoiding known allergens\n• Getting adequate rest and maintaining good health",
        
        treatment: "Treatment options include:\n• Artificial tears for comfort\n• Cold/warm compresses\n• Antibiotic drops (for bacterial infections)\n• Antiviral medications (for viral cases)\n• Regular cleaning of the affected area\n• Antihistamine drops for allergic reactions\n• Over-the-counter pain relievers\n\nDuration and Recovery:\n• Viral: 7-14 days\n• Bacterial: 5-7 days with antibiotics\n• Allergic: Improves with allergen removal",
        
        complications: "Possible complications if untreated:\n• Corneal inflammation\n• Vision problems\n• Chronic conjunctivitis\n• Spread to other eye\n• Secondary infections",
        
        types: "Different types of eye flu:\n\n1. Viral Conjunctivitis:\n• Most common type\n• Highly contagious\n• Often with common cold\n\n2. Bacterial Conjunctivitis:\n• Yellow/green discharge\n• More severe symptoms\n• Needs antibiotic treatment\n\n3. Allergic Conjunctivitis:\n• Seasonal occurrence\n• Affects both eyes\n• Intense itching",
        
        diagnosis: "Our AI-powered diagnosis process:\n• Image analysis of eye condition\n• Symptom pattern recognition\n• Medical history consideration\n• Severity assessment\n• Treatment recommendations"
    },
    
    services: {
        aiDiagnosis: "Our AI-powered system offers:\n• Instant eye condition analysis\n• High accuracy detection (95%+)\n• Detailed severity assessment\n• Custom treatment recommendations\n• Progress tracking\n• Early warning for complications",
        
        consultation: "Virtual consultation features:\n• 24/7 availability\n• Expert ophthalmologists\n• Personalized treatment plans\n• Follow-up care\n• Prescription services\n• Emergency care guidance",
        
        monitoring: "Recovery monitoring includes:\n• Daily progress tracking\n• Symptom improvement analysis\n• Treatment effectiveness assessment\n• Complication prevention\n• Recovery timeline updates\n• Automated reminders",
        
        technology: "Our technology features:\n• Advanced machine learning\n• Real-time analysis\n• Secure data handling\n• Mobile accessibility\n• Integration with medical records\n• Regular AI model updates"
    },
    
    website: {
        features: "Our website offers:\n• AI-powered eye flu detection\n• Symptom tracking dashboard\n• Virtual consultations\n• Treatment recommendations\n• Recovery monitoring\n• Educational resources\n• Emergency care guidance\n• Medication reminders\n• Progress reports",
        
        about: "EyeAI combines:\n• Advanced artificial intelligence\n• Medical expertise\n• Real-time analysis\n• Personalized care\n• Continuous monitoring\n• Data security\n\nOur mission is to make expert eye care accessible to everyone through innovative technology.",
        
        process: "Our process is simple:\n1. Upload an eye image\n2. Get AI analysis (30 seconds)\n3. Receive detailed diagnosis\n4. Get treatment recommendations\n5. Track your recovery\n6. Access follow-up care",
        
        security: "We ensure your privacy:\n• HIPAA compliance\n• Encrypted data storage\n• Secure transmission\n• Anonymous analysis\n• Regular security audits\n• Data protection protocols",
        
        support: "Support options:\n• 24/7 chat assistance\n• Email support\n• Video consultations\n• Emergency helpline\n• Resource center\n• Community forum"
    },
    
    emergencySigns: "Seek immediate medical attention if you experience:\n• Severe eye pain\n• Significant vision changes\n• Chemical exposure\n• Eye injury\n• Severe light sensitivity\n• Persistent symptoms\n\nOur emergency line: [Emergency Contact]",
    
    prevention: {
        lifestyle: "Lifestyle tips for eye health:\n• Regular hand washing\n• Balanced diet rich in vitamins A and C\n• Adequate sleep\n• Regular exercise\n• UV protection\n• Screen breaks (20-20-20 rule)",
        
        workplace: "Workplace eye protection:\n• Proper lighting\n• Ergonomic setup\n• Regular breaks\n• Protective eyewear\n• Clean environment\n• Air quality management",
        
        children: "Protecting children's eyes:\n• Regular check-ups\n• Limited screen time\n• Outdoor activities\n• Proper hygiene education\n• Protective eyewear during sports"
    }
};

// Expanded Knowledge Areas
const medicalKnowledge = {
    eyeAnatomy: {
        parts: "The eye consists of:\n• Cornea - Clear front layer\n• Iris - Colored part that controls light\n• Pupil - Dark center opening\n• Lens - Focuses light rays\n• Retina - Light-sensitive tissue\n• Optic nerve - Carries signals to brain\n• Conjunctiva - Membrane covering white part",
        
        functions: "Key eye functions:\n• Light processing\n• Focus adjustment\n• Color detection\n• Depth perception\n• Movement tracking\n• Environmental adaptation\n• Visual signal transmission"
    },
    
    commonEyeConditions: {
        dryEye: "Dry Eye Syndrome:\n• Insufficient tear production\n• Symptoms: burning, irritation\n• Causes: aging, environment, screen use\n• Treatment: artificial tears, lifestyle changes",
        
        glaucoma: "Glaucoma:\n• Increased eye pressure damaging optic nerve\n• Often no early symptoms\n• Can lead to vision loss if untreated\n• Regular screening recommended",
        
        cataracts: "Cataracts:\n• Clouding of the eye's lens\n• Symptoms: blurry vision, glare sensitivity\n• Common with aging\n• Treatment: surgical replacement",
        
        ageRelated: "Age-related Macular Degeneration:\n• Affects central vision\n• Leading cause of vision loss in older adults\n• Types: dry (more common) and wet\n• Management: nutrition, lifestyle, injections"
    },
    
    pediatricEyeHealth: {
        development: "Children's eye development:\n• Newborns: limited focus, can see 8-12 inches\n• 3-5 months: color vision develops\n• 8-12 months: depth perception improves\n• 3-5 years: visual acuity matures",
        
        common: "Common pediatric eye issues:\n• Amblyopia (lazy eye)\n• Strabismus (crossed eyes)\n• Refractive errors\n• Color blindness\n• Conjunctivitis (pink eye)",
        
        screenTime: "Screen time guidelines:\n• Under 2 years: avoid screen time\n• 2-5 years: 1 hour maximum\n• 6+ years: consistent limits\n• Frequent breaks using 20-20-20 rule"
    },
    
    eyeFluInDepth: {
        immune: "Immune response in eye flu:\n• Inflammatory reaction\n• Increased blood flow\n• White blood cell activation\n• Capillary dilation\n• Cytokine release",
        
        transmission: "Transmission pathways:\n• Direct contact with infected secretions\n• Contaminated surfaces and objects\n• Respiratory droplets\n• Swimming pools (rare)\n• Shared personal items",
        
        recovery: "Recovery process:\n• Inflammatory reduction\n• Tear film restoration\n• Epithelial cell regeneration\n• Microbiome rebalancing\n• Visual function normalization"
    },
    
    preventiveEyeCare: {
        screenings: "Recommended eye screenings:\n• Children: At birth, 6-12 months, 3-5 years\n• Adults 20-39: Every 5-10 years\n• Adults 40-54: Every 2-4 years\n• Adults 55+: Every 1-3 years\n• More frequent with risk factors",
        
        digitalHealth: "Digital eye health:\n• Blue light filters\n• Proper screen positioning\n• Adequate lighting\n• Regular breaks\n• Proper viewing distance\n• Ergonomic adjustments",
        
        occupational: "Occupation-specific protection:\n• Construction: Impact-resistant eyewear\n• Healthcare: Splash protection\n• Manufacturing: Chemical guards\n• Outdoor work: UV protection\n• Office: Anti-glare screens"
    }
};

const advancedFeatures = {
    symptomChecker: {
        questions: [
            "Is there redness in one or both eyes?",
            "Do you have any discharge from your eyes?",
            "Are your eyes itchy or burning?",
            "Do you have sensitivity to light?",
            "Is your vision affected in any way?",
            "Have you been in contact with someone with similar symptoms?",
            "Do you wear contact lenses?",
            "Have you recently been swimming or in a hot tub?"
        ],
        
        analyze: function(responses) {
            // Simulated analysis logic
            let viralScore = 0, bacterialScore = 0, allergicScore = 0;
            
            if (responses[0] === true) { // Redness
                viralScore += 1;
                bacterialScore += 1;
                allergicScore += 1;
            }
            
            if (responses[1] === true) { // Discharge
                bacterialScore += 2;
                viralScore += 1;
            }
            
            if (responses[2] === true) { // Itching
                allergicScore += 2;
            }
            
            // More analysis logic
            
            // Determine most likely type
            if (allergicScore > viralScore && allergicScore > bacterialScore) {
                return "allergic";
            } else if (bacterialScore > viralScore) {
                return "bacterial";
            } else {
                return "viral";
            }
        },
        
        recommendations: {
            viral: "Based on your symptoms, you may have viral conjunctivitis. Recommended steps:\n• Rest and let it run its course\n• Use cold compresses\n• Artificial tears for comfort\n• Avoid contact with others\n• Wash hands frequently",
            
            bacterial: "Based on your symptoms, you may have bacterial conjunctivitis. Recommended steps:\n• Consult a doctor for antibiotic treatment\n• Keep eyes clean\n• Avoid sharing personal items\n• Complete full course of antibiotics",
            
            allergic: "Based on your symptoms, you may have allergic conjunctivitis. Recommended steps:\n• Identify and avoid allergens\n• Try over-the-counter antihistamine drops\n• Cool compresses\n• Consider allergy medication"
        }
    },
    
    personalizedAdvice: {
        ageGroups: {
            child: "Children-specific advice:\n• Regular eye exams\n• Limited screen time\n• Protective eyewear during sports\n• Balanced diet rich in vitamins A and C\n• Proper lighting for reading",
            
            adult: "Adult-focused advice:\n• Regular breaks from screens\n• UV protection outdoors\n• Adequate sleep\n• Stay hydrated\n• Regular eye exams\n• Smoking cessation",
            
            senior: "Senior eye health tips:\n• Regular comprehensive exams\n• Monitor for age-related changes\n• Proper lighting\n• Fall prevention\n• Nutritional supplements if recommended"
        },
        
        riskGroups: {
            diabetic: "Advice for diabetic patients:\n• More frequent eye exams\n• Strict blood sugar control\n• Watch for vision changes\n• Regular retinal screening\n• Prompt attention to any issues",
            
            contact: "For contact lens wearers:\n• Strict hygiene practices\n• Follow replacement schedule\n• Never sleep in lenses\n• Keep backup glasses\n• Use preservative-free solutions if sensitive"
        }
    }
};

// Expand the function scope to handle more complex queries and use advanced features
function generateResponse(userMessage) {
    const message = userMessage.toLowerCase();
    chatContext.previousQuestions.push(message);
    
    // Update conversation context
    updateContext(message);
    
    // Emergency check
    if (isEmergency(message)) {
        return getEmergencyResponse();
    }
    
    // Check for symptom checker initiation
    if (message.includes("check my symptoms") || message.includes("symptom checker") || message.includes("diagnose")) {
        chatContext.currentTopic = "symptomChecker";
        chatContext.checkerState = {
            active: true,
            currentQuestion: 0,
            responses: []
        };
        
        return "I can help check your symptoms. Let's start with a few questions to better understand your condition.\n\n" +
               "First question: " + advancedFeatures.symptomChecker.questions[0] + "\n\n" +
               "(Please answer with yes or no)";
    }
    
    // Handle symptom checker conversation flow
    if (chatContext.currentTopic === "symptomChecker" && chatContext.checkerState?.active) {
        if (message.includes("yes") || message.includes("no")) {
            const response = message.includes("yes");
            chatContext.checkerState.responses.push(response);
            
            // Move to next question or provide result
            if (chatContext.checkerState.currentQuestion < advancedFeatures.symptomChecker.questions.length - 1) {
                chatContext.checkerState.currentQuestion++;
                return "Next question: " + advancedFeatures.symptomChecker.questions[chatContext.checkerState.currentQuestion] + "\n\n(Please answer with yes or no)";
            } else {
                // Analyze responses and provide recommendation
                const result = advancedFeatures.symptomChecker.analyze(chatContext.checkerState.responses);
                chatContext.checkerState.active = false;
                return advancedFeatures.symptomChecker.recommendations[result];
            }
        }
    }
    
    // Check for age-specific advice
    if (message.includes("advice") || message.includes("tips")) {
        if (message.includes("child") || message.includes("kid")) {
            return advancedFeatures.personalizedAdvice.ageGroups.child;
        } else if (message.includes("senior") || message.includes("elder") || message.includes("old")) {
            return advancedFeatures.personalizedAdvice.ageGroups.senior;
        } else if (message.includes("adult")) {
            return advancedFeatures.personalizedAdvice.ageGroups.adult;
        }
    }
    
    // Check for risk group specific advice
    if (message.includes("diabetes") || message.includes("diabetic")) {
        return advancedFeatures.personalizedAdvice.riskGroups.diabetic;
    } else if (message.includes("contact lens") || message.includes("contacts")) {
        return advancedFeatures.personalizedAdvice.riskGroups.contact;
    }
    
    // Check for eye anatomy questions
    if (message.includes("anatomy") || message.includes("structure") || message.includes("parts of the eye")) {
        return medicalKnowledge.eyeAnatomy.parts;
    } else if (message.includes("function") && (message.includes("eye") || message.includes("vision"))) {
        return medicalKnowledge.eyeAnatomy.functions;
    }
    
    // Check for other eye conditions
    if (message.includes("dry eye") || message.includes("dry eyes")) {
        return medicalKnowledge.commonEyeConditions.dryEye;
    } else if (message.includes("glaucoma")) {
        return medicalKnowledge.commonEyeConditions.glaucoma;
    } else if (message.includes("cataract")) {
        return medicalKnowledge.commonEyeConditions.cataracts;
    } else if ((message.includes("macular") || message.includes("amd")) || 
              (message.includes("age") && message.includes("degeneration"))) {
        return medicalKnowledge.commonEyeConditions.ageRelated;
    }
    
    // Check for pediatric eye health
    if ((message.includes("child") || message.includes("kid") || message.includes("baby")) && 
        (message.includes("eye") || message.includes("vision"))) {
        if (message.includes("development") || message.includes("grow")) {
            return medicalKnowledge.pediatricEyeHealth.development;
        } else if (message.includes("issue") || message.includes("problem")) {
            return medicalKnowledge.pediatricEyeHealth.common;
        } else if (message.includes("screen") || message.includes("device")) {
            return medicalKnowledge.pediatricEyeHealth.screenTime;
        } else {
            // General pediatric eye health
            return "Children's eye health is crucial. I can provide information about:\n" +
                   "• Eye development stages\n" +
                   "• Common childhood eye issues\n" +
                   "• Screen time guidelines\n\n" +
                   "What specific aspect would you like to know more about?";
        }
    }
    
    // Check for preventive eye care
    if (message.includes("prevent") && (message.includes("eye") || message.includes("vision"))) {
        if (message.includes("screen") || message.includes("exam")) {
            return medicalKnowledge.preventiveEyeCare.screenings;
        } else if (message.includes("digital") || message.includes("computer") || message.includes("screen")) {
            return medicalKnowledge.preventiveEyeCare.digitalHealth;
        } else if (message.includes("work") || message.includes("job") || message.includes("occupation")) {
            return medicalKnowledge.preventiveEyeCare.occupational;
        }
    }
    
    // Fall back to the previous response generation for other queries
    return getContextualResponse(message);
}

// Enhance the sendMessage function to handle the symptom checker
function sendMessage() {
    const input = document.getElementById('chatInput');
    const message = input.value.trim();
    
    if (message) {
        addMessage(message, 'user');
        input.value = '';
        
        // Auto-resize input
        input.style.height = 'auto';
        
        // Generate response based on user input and context
        setTimeout(() => {
            const response = generateResponse(message);
            addMessage(response, 'bot');
        }, 1000);
    }
}

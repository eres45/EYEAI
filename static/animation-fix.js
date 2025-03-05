// Fix for animations
document.addEventListener('DOMContentLoaded', function() {
    // Fix for stat animations
    const statItems = document.querySelectorAll('.stat-item');
    
    // Manually trigger the stat animation
    const statsObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('active');
                // Call the counter animation
                animateCounter(entry.target);
            }
        });
    }, { threshold: 0.3 });
    
    statItems.forEach(stat => statsObserver.observe(stat));
    
    // Improved counter animation function
    function animateCounter(statItem) {
        const numberElement = statItem.querySelector('.stat-number');
        if (!numberElement) return;
        
        // Get the target value from the data-value attribute
        const finalValue = parseInt(statItem.dataset.value);
        if (isNaN(finalValue)) return;
        
        // Special case for availability (24/7)
        const statLabel = statItem.querySelector('.stat-label');
        if (statLabel && statLabel.textContent.includes('Availability')) {
            numberElement.textContent = '0/7';
            setTimeout(() => {
                numberElement.textContent = '24/7';
            }, 1000);
            return;
        }
        
        // Get suffix (%, s, k+, etc.)
        let suffix = '';
        if (numberElement.textContent.includes('%')) suffix = '%';
        if (numberElement.textContent.includes('s')) suffix = 's';
        if (numberElement.textContent.includes('k+')) suffix = 'k+';
        if (numberElement.textContent.includes('+')) suffix = '+';
        
        // Animation variables
        let currentValue = 0;
        const duration = 2000; // 2 seconds
        const steps = 60;
        const increment = finalValue / steps;
        const stepDuration = duration / steps;
        
        // Start from 0
        numberElement.textContent = '0' + suffix;
        
        // Clear any existing animation
        if (statItem.counterInterval) {
            clearInterval(statItem.counterInterval);
        }
        
        // Create the counting animation
        statItem.counterInterval = setInterval(() => {
            currentValue += increment;
            
            if (currentValue >= finalValue) {
                currentValue = finalValue;
                clearInterval(statItem.counterInterval);
                statItem.counterInterval = null;
            }
            
            numberElement.textContent = Math.floor(currentValue) + suffix;
        }, stepDuration);
    }
    
    // Fix for spiral diagram
    const spiralElement = document.querySelector('.process-spiral');
    if (spiralElement) {
        const spiralObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    // Force add active class to spiral
                    spiralElement.classList.add('active');
                    
                    // Reset spiral animation
                    const spiralPaths = spiralElement.querySelectorAll('.spiral-path');
                    spiralPaths.forEach(path => {
                        path.style.animation = 'none';
                        setTimeout(() => {
                            path.style.animation = 'drawCircle 2s cubic-bezier(0.4, 0, 0.2, 1) forwards';
                        }, 100);
                    });
                    
                    // Animate the spiral points
                    const spiralPoints = spiralElement.querySelectorAll('.spiral-point');
                    spiralPoints.forEach((point, index) => {
                        setTimeout(() => {
                            point.style.opacity = '1';
                            const angle = point.style.getPropertyValue('--angle');
                            point.style.transform = `
                                rotate(${angle})
                                translateX(300px)
                                rotate(calc(-1 * ${angle}))
                            `;
                        }, 500 + (index * 150));
                    });
                }
            });
        }, { threshold: 0.3 });
        
        spiralObserver.observe(spiralElement);
    }
});

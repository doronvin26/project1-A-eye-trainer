const DebugUtils = {
    overlay: document.getElementById('debug-overlay'),

    log(message) {
        console.log(`[DEBUG]: ${message}`);
        const div = document.createElement('div');
        div.innerText = message;
        this.overlay.prepend(div);
        
        // הגבלה של 5 לוגים אחרונים כדי לא להעמיס
        if (this.overlay.children.length > 5) {
            this.overlay.removeChild(this.overlay.lastChild);
        }
    },

    // פונקציה למדידת זמן ריצה (בדומה ל-Times(ms) שהיה לך)
    updateMetrics(metrics) {
        const text = Object.entries(metrics)
            .map(([key, val]) => `${key}: ${val.toFixed(1)}ms`)
            .join(' | ');
        this.log(`Times -> ${text}`);
    }
};
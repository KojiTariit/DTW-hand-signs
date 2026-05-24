export function initUI() {
    const tabs = document.querySelectorAll('.nav-tab');
    const views = {
        'view-translate': document.getElementById('view-translate'),
        'view-history': document.getElementById('view-history'),
        'view-settings': document.getElementById('view-settings')
    };
    
    // Header History shortcut
    document.getElementById('btn-header-history').addEventListener('click', () => switchView('view-history'));

    tabs.forEach(tab => {
        tab.addEventListener('click', (e) => {
            const target = e.currentTarget.getAttribute('data-target');
            if (target) switchView(target);
        });
    });

    function switchView(targetId) {
        // Hide all
        Object.values(views).forEach(v => {
            if(v) {
                v.classList.remove('opacity-100', 'z-10');
                v.classList.add('opacity-0', 'pointer-events-none', 'z-0');
            }
        });

        // Show target
        if (views[targetId]) {
            views[targetId].classList.remove('opacity-0', 'pointer-events-none', 'z-0');
            views[targetId].classList.add('opacity-100', 'z-10');
        }

        // Update tabs styling
        tabs.forEach(tab => {
            const t = tab.getAttribute('data-target');
            if (t === targetId) {
                tab.classList.remove('text-slate-400');
                tab.classList.add('text-brand-blue');
                tab.querySelector('div')?.classList.add('bg-brand-blue', 'text-white');
                tab.querySelector('i')?.classList.add('ph-fill');
            } else {
                tab.classList.remove('text-brand-blue');
                tab.classList.add('text-slate-400');
                tab.querySelector('div')?.classList.remove('bg-brand-blue', 'text-white');
                tab.querySelector('i')?.classList.remove('ph-fill');
            }
        });
    }
}

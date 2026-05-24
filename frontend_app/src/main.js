import '../style.css'
import { initUI } from './ui.js'
import { CameraEngine } from './camera.js'

document.addEventListener('DOMContentLoaded', () => {
  // 1. Initialize UI Interactions
  initUI();

  // 2. Initialize Camera and Engine
  const engine = new CameraEngine();
  engine.initialize();
});

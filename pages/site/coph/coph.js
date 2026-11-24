// The poster's chapter buttons: each seeks the walkthrough to its data-t (seconds) and plays it.
document.addEventListener('DOMContentLoaded', () => {
  const video = document.getElementById('walkthrough');
  if (!video) return;
  for (const b of document.querySelectorAll('.chapters button[data-t]')) {
    b.addEventListener('click', () => {
      video.currentTime = Number(b.dataset.t) || 0;
      video.play().catch(() => {});
    });
  }
});

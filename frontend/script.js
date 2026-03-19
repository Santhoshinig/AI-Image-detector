const imageInput = document.getElementById('imageInput');
const dropzone = document.getElementById('dropzone');
const previewBox = document.getElementById('previewBox');
const analyzeBtn = document.getElementById('analyzeBtn');
const resetBtn = document.getElementById('resetBtn');
const statusText = document.getElementById('statusText');
const loader = document.getElementById('loader');
const resultEmpty = document.getElementById('resultEmpty');
const resultContent = document.getElementById('resultContent');

let selectedFile = null;

function setPreview(file) {
  const reader = new FileReader();
  reader.onload = e => {
    previewBox.innerHTML = `<img src="${e.target.result}" alt="Selected image preview" />`;
  };
  reader.readAsDataURL(file);
}

function updateSelectedFile(file) {
  selectedFile = file;
  analyzeBtn.disabled = !file;

  if (!file) {
    previewBox.innerHTML = `<p class="empty-preview">Your selected image will appear here.</p>`;
    statusText.textContent = 'No image selected.';
    return;
  }

  statusText.textContent = `Selected: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
  setPreview(file);
}

imageInput.addEventListener('change', e => {
  const file = e.target.files[0];
  if (file) updateSelectedFile(file);
});

dropzone.addEventListener('dragover', e => {
  e.preventDefault();
  dropzone.classList.add('dragover');
});

dropzone.addEventListener('dragleave', () => {
  dropzone.classList.remove('dragover');
});

dropzone.addEventListener('drop', e => {
  e.preventDefault();
  dropzone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file) {
    imageInput.files = e.dataTransfer.files;
    updateSelectedFile(file);
  }
});

resetBtn.addEventListener('click', () => {
  selectedFile = null;
  imageInput.value = '';
  updateSelectedFile(null);
  resultEmpty.classList.remove('hidden');
  resultContent.classList.add('hidden');
});

analyzeBtn.addEventListener('click', async () => {
  if (!selectedFile) return;

  loader.classList.remove('hidden');
  analyzeBtn.disabled = true;
  statusText.textContent = 'Analyzing image...';

  const formData = new FormData();
  formData.append('image', selectedFile);

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || 'Something went wrong while analyzing the image.');
    }

    renderResult(data);
    statusText.textContent = 'Analysis complete.';
  } catch (error) {
    alert(error.message);
    statusText.textContent = 'Analysis failed.';
  } finally {
    loader.classList.add('hidden');
    analyzeBtn.disabled = false;
  }
});

function renderResult(data) {
  resultEmpty.classList.add('hidden');
  resultContent.classList.remove('hidden');

  document.getElementById('labelPill').textContent = data.label;
  document.getElementById('confidencePill').textContent = `AI: ${data.ai_probability}%`;
  document.getElementById('aiProbPill').textContent = `Real: ${data.real_probability}%`;
  document.getElementById('noteBox').textContent = data.note;

  document.getElementById('widthValue').textContent = data.details.width;
  document.getElementById('heightValue').textContent = data.details.height;
  document.getElementById('megaValue').textContent = data.details.megapixels;
  document.getElementById('textureValue').textContent = data.details.texture_std;
  document.getElementById('edgeValue').textContent = data.details.edge_density;

  const reasonsList = document.getElementById('reasonsList');
  reasonsList.innerHTML = '';
  data.details.reasons.forEach(reason => {
    const li = document.createElement('li');
    li.textContent = reason;
    reasonsList.appendChild(li);
  });
}

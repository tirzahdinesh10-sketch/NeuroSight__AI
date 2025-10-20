const fileInput = document.getElementById('fileInput');
const browseBtn = document.getElementById('browseBtn');
const previewBox = document.getElementById('previewBox');
const tumorTypeEl = document.getElementById('tumorType');
const confidenceEl = document.getElementById('confidence');
const metricsSummary = document.getElementById('metricsSummary');

browseBtn.addEventListener('click', ()=> fileInput.click());
fileInput.addEventListener('change', async (e)=>{
  const f = e.target.files[0];
  if(!f) return;
  previewBox.innerHTML = '';
  const img = document.createElement('img');
  img.src = URL.createObjectURL(f);
  img.style.maxHeight = '200px';
  img.onload = ()=> URL.revokeObjectURL(img.src);
  previewBox.appendChild(img);

  // upload
  const form = new FormData();
  form.append('file', f);
  const res = await fetch('/predict', {method:'POST', body: form});
  if(!res.ok){
    const err = await res.json();
    tumorTypeEl.textContent = 'Error';
    confidenceEl.textContent = err.error || 'failed';
    return;
  }
  const data = await res.json();
  tumorTypeEl.textContent = data.prediction;
  if(data.probabilities && data.classes){
    const idx = data.classes.indexOf(data.prediction);
    confidenceEl.textContent = idx>=0? (Math.round((data.probabilities[idx]*100)*10)/10)+'%':'-';
    showChart(data.classes, data.probabilities);
  }
});

let chart=null;
function showChart(labels, probs){
  const ctx = document.getElementById('metricsChart').getContext('2d');
  if(chart) chart.destroy();
  chart = new Chart(ctx, {
    type:'bar',
    data:{labels:labels, datasets:[{label:'probability',data:probs,backgroundColor:['#4f46e5','#06b6d4','#60a5fa','#34d399']} ]},
    options:{scales:{y:{min:0,max:1}}}
  });
}

// initial metrics placeholder
metricsSummary.textContent = 'Model metrics will appear here after prediction.';

// metrics loading removed — no test accuracy or confusion image displayed by frontend

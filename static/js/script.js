// ---------------- theme toggle -----------------
const html = document.documentElement,
      tg   = document.getElementById('themeTg');

if (tg){
  const pref = localStorage.getItem('bsTheme');
  const sysDark = window.matchMedia('(prefers-color-scheme:dark)').matches;
  const initMode = pref || (sysDark ? 'dark':'light');
  html.dataset.bsTheme = initMode;
  tg.checked = initMode === 'dark';

  tg.onchange = e=>{
    const m = e.target.checked ? 'dark':'light';
    html.dataset.bsTheme = m;
    localStorage.setItem('bsTheme', m);
  };
}

// -------------- input label auto-detect ----------
const fld = document.getElementById("text-input"),
      lbl = document.getElementById("input-label"),
      frm = document.getElementById("predict-form");

function isURL(t){ return /^(https?:\/\/)?[^\s]+\.[^\s]+/.test(t); }

fld?.addEventListener("input", ()=>{
  const v = fld.value.trim();
  lbl.textContent = v ? (isURL(v)?"Detected URL":"Detected Email Text")
                      : "Enter URL or Email Text";
});

frm?.addEventListener("submit", e=>{
  if (!fld.value.trim()){ e.preventDefault(); alert("Please enter some text 🙂"); }
});

// -------------- fade-up intersection -----------
document.querySelectorAll('.fade-up').forEach(el=>{
  const io=new IntersectionObserver(([e])=>{
    if(e.isIntersecting){ e.target.classList.add('show'); io.disconnect();}
  });
  io.observe(el);
});

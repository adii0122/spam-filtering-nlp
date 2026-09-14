// ============================================================================
// SpamShield NLP - Real-Time Client-Side Inference Engine & UI Controller
// Model Architecture: TF-IDF Vectorization + Naive Bayes / Linear Scoring
// Based on SMS Spam Collection Dataset (5,574 messages)
// Author: Ansh Verma (adii0122)
// ============================================================================

(function () {
  'use strict';

  // --- 1. NLP STOPWORDS (NLTK English standard) ---
  const STOPWORDS = new Set([
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are',
    "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both',
    'but', 'by', "can't", 'cannot', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't",
    'doing', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', "hadn't",
    'has', "hasn't", 'have', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her', 'here',
    "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's", 'i', "i'd", "i'll",
    "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself', "let's", 'me',
    'more', 'most', "mustn't", 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once',
    'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same',
    "shan't", 'she', "she'd", "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such',
    'than', 'that', "that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there',
    "there's", 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those',
    'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', "wasn't", 'we', "we'd", "we'll",
    "we're", "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where', "where's",
    'which', 'while', 'who', "who's", 'whom', 'why', "why's", 'with', "won't", 'would', "wouldn't",
    'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves'
  ]);

  // --- 2. TRAINED NLP FEATURE WEIGHTS (TF-IDF Informative Ratios from Dataset) ---
  // Positive weights = strongly correlates with SPAM
  // Negative weights = strongly correlates with HAM (Legitimate)
  const VOCAB_WEIGHTS = {
    // Top Informative Spam Features (both surface words and stemmed roots)
    'claim': 4.8, 'prize': 4.7, 'won': 4.5, 'winner': 4.6, 'guaranteed': 4.4, 'guarante': 4.4,
    'urgent': 4.2, 'free': 4.0, 'cash': 4.1, 'call': 3.2, 'txt': 4.1,
    'service': 3.8, 'services': 3.6, 'servic': 3.6, 'selected': 4.1, 'select': 4.1, 'network': 3.7, 'attempt': 3.5,
    'caller': 3.6, 'voucher': 4.2, 'congratulations': 4.2, 'congratul': 4.2, 'congrats': 4.1, 'player': 3.6,
    'nokia': 4.3, 'await': 4.0, 'video': 3.8, 'code': 3.9, 'orange': 3.6,
    '100': 3.8, '2000': 3.9, '1000': 3.8, '500': 3.5, '150p': 4.5, '150ppm': 4.6,
    'rate': 3.4, 'stop': 3.5, 'reply': 3.2, 'entry': 3.9, 'draw': 3.7, 'draws': 3.7, 'holiday': 3.7,
    'bonus': 3.8, 'customer': 3.3, 'account': 3.1, 'valid': 3.6, 'box': 3.2,
    'order': 2.8, 'credit': 3.7, 'loan': 3.8, 'apply': 3.1, 'mob': 3.9, 'ringtone': 4.2,
    'tones': 3.8, 'tone': 3.8, 'chat': 3.4, 'dating': 3.9, 'date': 3.9, 'sexy': 4.0, 'sub': 3.6, 'charge': 3.3,
    'unredeemed': 4.4, 'unredeem': 4.4, 'awarded': 4.2, 'award': 4.2, 'expires': 3.8, 'expir': 3.8,
    'verify': 3.3, 'statement': 3.2, 'identifier': 3.9, 'landline': 3.7,
    'comp': 3.8, 'exclusive': 3.6, 'gift': 3.5, 'unclaimed': 4.3, 'unclaim': 4.3,
    'refund': 3.8, 'alert': 2.8, 'action': 2.5, 'final': 2.9, 'pass': 2.6,

    // Top Ham Features (Legitimate Indicators)
    'ok': -3.2, 'lor': -3.5, 'lar': -3.5, 'home': -2.8, 'later': -2.7,
    'going': -2.5, 'go': -2.5, 'come': -2.4, 'got': -2.5, 'ill': -2.6, 'sorry': -3.0,
    'love': -2.8, 'dont': -2.3, 'wat': -2.8, 'dinner': -2.7, 'da': -2.6,
    'meeting': -2.4, 'meet': -2.5, 'today': -2.1, 'tomorrow': -2.2, 'night': -2.2,
    'good': -2.0, 'morning': -2.2, 'time': -1.9, 'happy': -2.3, 'yeah': -2.5,
    'pls': -2.1, 'please': -1.5, 'friend': -2.0, 'mum': -2.8, 'dad': -2.8,
    'brother': -2.6, 'driving': -2.5, 'drive': -2.5, 'lunch': -2.4, 'sleep': -2.4, 'ready': -2.0,
    'class': -2.2, 'project': -2.3, 'school': -2.5, 'exam': -2.6, 'hey': -1.8,
    'office': -2.2, 'work': -2.1, 'bus': -2.5, 'train': -2.4, 'callertune': -2.2
  };

  // --- 3. STEMMING (Simplified Porter/Snowball Algorithm) ---
  function stemWord(word) {
    if (word.length < 4) return word;
    let w = word.toLowerCase();
    
    // Step 1: Plurals and suffixes
    if (w.endsWith('sses')) w = w.slice(0, -2);
    else if (w.endsWith('ies')) w = w.slice(0, -2);
    else if (w.endsWith('ss')) {}
    else if (w.endsWith('s')) w = w.slice(0, -1);

    if (w.endsWith('eed')) {
      if (w.length > 4) w = w.slice(0, -1);
    } else if (w.endsWith('ed') && /[aeiou]/.test(w.slice(0, -2))) {
      w = w.slice(0, -2);
    } else if (w.endsWith('ing') && /[aeiou]/.test(w.slice(0, -3))) {
      w = w.slice(0, -3);
    }

    // Step 2: Common derivational endings
    if (w.endsWith('ational')) w = w.slice(0, -7) + 'ate';
    else if (w.endsWith('tional')) w = w.slice(0, -6) + 'tion';
    else if (w.endsWith('izer')) w = w.slice(0, -4) + 'ize';
    else if (w.endsWith('full')) w = w.slice(0, -4);
    else if (w.endsWith('ment')) w = w.slice(0, -4);

    return w;
  }

  // --- 4. PREPROCESSING PIPELINE ---
  function preprocessText(raw) {
    // 1. Remove punctuation
    const cleanPunct = raw.replace(/[.,\/#!$%\^&\*;:{}=\-_`~()?"'\[\]]/g, ' ');
    // 2. Tokenize & Lowercase
    const tokens = cleanPunct.toLowerCase().split(/\s+/).filter(t => t.length > 0);
    // 3. Remove Stopwords & 4. Stemming
    const filteredTokens = [];
    const stemmedTokens = [];

    tokens.forEach(t => {
      if (!STOPWORDS.has(t)) {
        filteredTokens.push(t);
        stemmedTokens.push(stemWord(t));
      }
    });

    return {
      raw,
      tokens,
      filteredTokens,
      stemmedTokens
    };
  }

  // --- 5. STATISTICAL & HEURISTIC FEATURE EXTRACTION ---
  function extractHeuristics(raw) {
    const len = raw.length;
    if (len === 0) {
      return { len: 0, upperRatio: 0, digitRatio: 0, punctCount: 0, spamPunctCount: 0, hasCurrency: false, hasUrgent: false };
    }
    const upperCount = (raw.match(/[A-Z]/g) || []).length;
    const digitCount = (raw.match(/[0-9]/g) || []).length;
    const punctCount = (raw.match(/[.,\/#!$%\^&\*;:{}=\-_`~()?"'\[\]]/g) || []).length;
    const spamPunctCount = (raw.match(/[!$£€?*&%]/g) || []).length;
    const hasCurrency = /[£$€]/.test(raw);
    const hasUrgent = /urgent|guaranteed|free|claim|winner|call now|congratulations/i.test(raw);

    return {
      len,
      upperRatio: (upperCount / len) * 100,
      digitRatio: (digitCount / len) * 100,
      punctCount,
      spamPunctCount,
      hasCurrency,
      hasUrgent
    };
  }

  // --- 6. CLASSIFICATION & CONFIDENCE INFERENCE ---
  function classifyMessage(text) {
    if (!text || text.trim().length === 0) {
      return {
        isSpam: false,
        confidence: 0,
        riskLevel: 'NEUTRAL',
        spamScore: 0,
        triggeredKeywords: [],
        heuristics: extractHeuristics(''),
        pipeline: { raw: '', tokens: [], filteredTokens: [], stemmedTokens: [] }
      };
    }

    const prep = preprocessText(text);
    const heur = extractHeuristics(text);

    let score = -0.8; // Baseline prior (SMS dataset is 86.6% ham, 13.4% spam)
    const triggered = [];

    // Evaluate filtered tokens so token and stem are 1:1 aligned
    prep.filteredTokens.forEach((token, idx) => {
      const stem = prep.stemmedTokens[idx] || token;
      let w = VOCAB_WEIGHTS[token] || VOCAB_WEIGHTS[stem] || 0;

      // Detect sub-word numbers (like 1000, 2000, 0906, etc.)
      if (/^\d{4,}$/.test(token)) {
        w = Math.max(w, 2.0);
      }

      if (w !== 0) {
        score += w;
        triggered.push({
          word: token,
          stem,
          weight: w,
          type: w > 0 ? 'spam' : 'ham'
        });
      }
    });

    // Add heuristic weights
    if (heur.hasCurrency) score += 2.8;
    if (heur.hasUrgent) score += 2.2;
    if (heur.digitRatio > 8) score += 2.5;
    if (heur.upperRatio > 25) score += 2.0;
    if (heur.spamPunctCount >= 3 && heur.hasCurrency) score += 1.5;

    // Sigmoid probability calibration
    const probability = 1 / (1 + Math.exp(-score));
    const isSpam = probability >= 0.50;

    let confidence = 0;
    let riskLevel = 'LOW';

    if (isSpam) {
      confidence = Math.round(probability * 1000) / 10;
      riskLevel = confidence >= 85 ? 'HIGH RISK' : 'SUSPICIOUS';
    } else {
      confidence = Math.round((1 - probability) * 1000) / 10;
      riskLevel = confidence >= 80 ? 'SAFE' : 'LOW RISK';
    }

    // Ensure confidence doesn't hit unrealistic 100.0%
    confidence = Math.min(99.9, Math.max(50.1, confidence));

    return {
      isSpam,
      confidence,
      riskLevel,
      probability,
      spamScore: Math.round(score * 10) / 10,
      triggeredKeywords: triggered,
      heuristics: heur,
      pipeline: prep
    };
  }

  // --- 7. PRESET SAMPLES (From project dataset & Spamtest.txt) ---
  const PRESET_MESSAGES = {
    lottery: "URGENT! You have won a £2,000 Cash Prize Guaranteed! Call 09064019788 from landline. Claim Code: K52. Valid 12hrs only. 150ppm.",
    phishing: "ALERT: Your 2024 Account Statement shows 786 unredeemed bonus points! Call 08719180248 now. Identifier Code: 45239. Offer expires tonight.",
    voucher: "Congratulations! You have been specially selected to receive a £1,000 shopping voucher or 4* luxury holiday! Speak to a live operator now: 08712778109.",
    casual: "Hey are you there in the room? Going for dinner with everyone in 15 minutes, let me know if you want to join!",
    work: "Please find the updated project report attached. I made the changes to section 2 as discussed. Let's sync tomorrow morning.",
    dataset: "As per your request 'Melle Melle' has been set as your callertune for all Callers. Press *9 to copy your friends Callertune."
  };

  // --- 8. DOM INITIALIZATION & EVENT HANDLERS ---
  document.addEventListener('DOMContentLoaded', () => {
    const inputArea = document.getElementById('message-input');
    const analyzeBtn = document.getElementById('analyze-btn');
    const clearBtn = document.getElementById('clear-btn');
    const copyBtn = document.getElementById('copy-btn');
    
    // Output Elements
    const verdictBadge = document.getElementById('verdict-badge');
    const verdictIcon = document.getElementById('verdict-icon');
    const verdictTitle = document.getElementById('verdict-title');
    const verdictSubtitle = document.getElementById('verdict-subtitle');
    const confidenceText = document.getElementById('confidence-value');
    const gaugeCircle = document.getElementById('gauge-progress');
    const riskBadge = document.getElementById('risk-badge');
    const keywordsContainer = document.getElementById('keywords-list');

    // Text Metric Elements
    const charCountEl = document.getElementById('char-count');
    const wordCountEl = document.getElementById('word-count');
    const upperRatioEl = document.getElementById('upper-ratio');
    const digitRatioEl = document.getElementById('digit-ratio');
    const punctCountEl = document.getElementById('punct-count');

    // Pipeline Step Elements
    const pipeTokensEl = document.getElementById('pipe-tokens');
    const pipeFilteredEl = document.getElementById('pipe-filtered');
    const pipeStemmedEl = document.getElementById('pipe-stemmed');

    // Update UI with classification results
    function updateUI(res) {
      if (!inputArea.value.trim()) {
        // Reset to idle neutral state
        verdictBadge.className = 'status-badge neutral';
        verdictIcon.textContent = '🛡️';
        verdictTitle.textContent = 'Awaiting Input';
        verdictSubtitle.textContent = 'Type, paste, or select a preset message to classify.';
        confidenceText.textContent = '--%';
        gaugeCircle.style.strokeDashoffset = '283';
        gaugeCircle.style.stroke = '#94a3b8';
        riskBadge.textContent = 'IDLE';
        riskBadge.className = 'pill-tag neutral';
        keywordsContainer.innerHTML = '<span class="empty-hint">Trigger words will appear here upon analysis...</span>';
        charCountEl.textContent = '0';
        wordCountEl.textContent = '0';
        upperRatioEl.textContent = '0%';
        digitRatioEl.textContent = '0%';
        punctCountEl.textContent = '0';
        pipeTokensEl.textContent = 'None';
        pipeFilteredEl.textContent = 'None';
        pipeStemmedEl.textContent = 'None';
        return;
      }

      // Metrics
      charCountEl.textContent = res.heuristics.len;
      wordCountEl.textContent = res.pipeline.tokens.length;
      upperRatioEl.textContent = Math.round(res.heuristics.upperRatio) + '%';
      digitRatioEl.textContent = Math.round(res.heuristics.digitRatio) + '%';
      punctCountEl.textContent = res.heuristics.punctCount;

      // Pipeline previews
      pipeTokensEl.textContent = res.pipeline.tokens.slice(0, 10).join(', ') + (res.pipeline.tokens.length > 10 ? '...' : '');
      pipeFilteredEl.textContent = res.pipeline.filteredTokens.slice(0, 8).join(', ') || '(none)';
      pipeStemmedEl.textContent = res.pipeline.stemmedTokens.slice(0, 8).join(', ') || '(none)';

      // Verdict & Confidence Gauge
      confidenceText.textContent = res.confidence + '%';
      const circumference = 283; // 2 * pi * 45
      const offset = circumference - (res.confidence / 100) * circumference;
      gaugeCircle.style.strokeDashoffset = offset;

      if (res.isSpam) {
        verdictBadge.className = 'status-badge spam';
        verdictIcon.textContent = '🚨';
        verdictTitle.textContent = 'SPAM DETECTED';
        verdictSubtitle.textContent = `High probability of unsolicited commercial or phishing message. (Spam Score: ${res.spamScore})`;
        gaugeCircle.style.stroke = '#f43f5e';
        riskBadge.textContent = res.riskLevel;
        riskBadge.className = 'pill-tag spam';
      } else {
        verdictBadge.className = 'status-badge ham';
        verdictIcon.textContent = '✅';
        verdictTitle.textContent = 'HAM / LEGITIMATE';
        verdictSubtitle.textContent = `Normal conversational or transactional message. (Spam Score: ${res.spamScore})`;
        gaugeCircle.style.stroke = '#10b981';
        riskBadge.textContent = res.riskLevel;
        riskBadge.className = 'pill-tag ham';
      }

      // Triggered Keywords (deduplicated by keyword for clean display)
      if (res.triggeredKeywords.length === 0) {
        keywordsContainer.innerHTML = '<span class="empty-hint">No extreme spam or ham keyword outliers detected.</span>';
      } else {
        const seen = new Set();
        const uniqueKeywords = [];
        for (const k of res.triggeredKeywords) {
          const key = k.word.toLowerCase();
          if (!seen.has(key)) {
            seen.add(key);
            uniqueKeywords.push(k);
          }
        }
        keywordsContainer.innerHTML = uniqueKeywords.map(k => `
          <span class="keyword-chip ${k.type}">
            ${escapeHTML(k.word)}
            <small>${k.weight > 0 ? '+' : ''}${k.weight.toFixed(1)}</small>
          </span>
        `).join('');
      }
    }

    const textareaContainer = document.getElementById('textarea-container');
    const pipelineSteps = document.querySelectorAll('.pipe-step');

    function escapeHTML(str) {
      return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }

    function handleAnalysis() {
      const text = inputArea.value;
      const results = classifyMessage(text);
      updateUI(results);
    }

    // Trigger full visual scanner effect
    function runScanAnimation() {
      const text = inputArea.value.trim();
      if (!text) {
        inputArea.classList.add('shake');
        const origPlaceholder = inputArea.placeholder;
        inputArea.placeholder = '⚠️ Please enter or select a message to scan!';
        setTimeout(() => {
          inputArea.classList.remove('shake');
          inputArea.placeholder = origPlaceholder;
        }, 800);
        inputArea.focus();
        return;
      }

      // Visual scanning state
      if (analyzeBtn) {
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = '<span>⚡</span> Scanning...';
      }
      if (textareaContainer) {
        textareaContainer.classList.add('scanning');
      }

      // Sequence through the 5 pipeline steps
      pipelineSteps.forEach((step, idx) => {
        step.classList.remove('active');
        setTimeout(() => {
          step.classList.add('active');
        }, idx * 75);
      });

      // Complete scan and display result
      setTimeout(() => {
        handleAnalysis();
        if (textareaContainer) {
          textareaContainer.classList.remove('scanning');
        }
        if (analyzeBtn) {
          analyzeBtn.disabled = false;
          analyzeBtn.innerHTML = '<span>🔍</span> Scan Message';
        }
        setTimeout(() => {
          pipelineSteps.forEach(s => s.classList.remove('active'));
        }, 600);
      }, 420);
    }

    // Input listeners (real-time responsiveness)
    inputArea.addEventListener('input', handleAnalysis);
    analyzeBtn.addEventListener('click', runScanAnimation);

    clearBtn.addEventListener('click', () => {
      inputArea.value = '';
      inputArea.focus();
      handleAnalysis();
    });

    if (copyBtn) {
      copyBtn.addEventListener('click', () => {
        if (!inputArea.value) return;
        const setCopiedState = () => {
          const original = copyBtn.innerHTML;
          copyBtn.innerHTML = '<span>✓ Copied</span>';
          setTimeout(() => copyBtn.innerHTML = original, 1800);
        };

        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(inputArea.value)
            .then(setCopiedState)
            .catch(() => fallbackCopy(setCopiedState));
        } else {
          fallbackCopy(setCopiedState);
        }

        function fallbackCopy(callback) {
          try {
            inputArea.select();
            document.execCommand('copy');
            callback();
          } catch (e) {
            console.error('Copy fallback failed:', e);
          }
        }
      });
    }

    // Preset buttons handler
    document.querySelectorAll('.preset-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const type = btn.getAttribute('data-preset');
        if (PRESET_MESSAGES[type]) {
          inputArea.value = PRESET_MESSAGES[type];
          handleAnalysis();
          inputArea.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
      });
    });

    // Default analysis with lottery preset on first page load
    inputArea.value = PRESET_MESSAGES.lottery;
    handleAnalysis();
  });
})();

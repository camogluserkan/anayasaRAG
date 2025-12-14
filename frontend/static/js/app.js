// Minimal inline SVG icon set (stroke-based, currentColor)
const Icons = {
  user: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>',
  scales: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v18"/><path d="M3 7h18"/><path d="M7 7L3 14a4 4 0 0 0 8 0L7 7z"/><path d="M17 7l-4 7a4 4 0 0 0 8 0l-4-7z"/></svg>',
  book: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M4 4v15.5A2.5 2.5 0 0 0 6.5 22H20V6a2 2 0 0 0-2-2H6"/></svg>',
  check: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6L9 17l-5-5"/></svg>',
  alert: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L14.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
  x: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>',
  chip: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="8" y="8" width="8" height="8"/></svg>'
};

// Application State
let isWaitingForResponse = false;
let messageHistory = [];

/**
 * Render Markdown text to HTML.
 * Uses 'marked' library if available, otherwise falls back to basic regex replacement.
 * @param {string} text - Markdown text
 * @returns {string} HTML string
 */
function renderMarkdown(text) {
    if (typeof marked !== 'undefined') {
        return marked.parse(text);
    }
    // Fallback: simple markdown-like formatting
    return text
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/^## (.*$)/gm, '<h2>$1</h2>')
        .replace(/^# (.*$)/gm, '<h1>$1</h1>')
        .replace(/^- (.*$)/gm, '<li>$1</li>')
        .replace(/^(\d+)\. (.*$)/gm, '<li>$1. $2</li>')
        .replace(/\n\n/g, '</p><p>')
        .replace(/^(.*)$/gm, '<p>$1</p>')
        .replace(/<p><\/p>/g, '')
        .replace(/<p>(<h[12]>.*<\/h[12]>)<\/p>/g, '$1')
        .replace(/<p>(<li>.*<\/li>)<\/p>/g, '<ul>$1</ul>')
        .replace(/<\/li><li>/g, '</li><li>');
}

// DOM Elements
const welcomeScreen = document.getElementById('welcomeScreen');
const messagesContainer = document.getElementById('messagesContainer');
const messageInput = document.getElementById('messageInput');
const sendButton = document.getElementById('sendButton');
const charCount = document.getElementById('charCount');
const systemStatus = document.getElementById('systemStatus');
const chunkCount = document.getElementById('chunkCount');
const pdfList = document.getElementById('pdfList');

// Initialize Application
document.addEventListener('DOMContentLoaded', () => {
    loadSystemStatus();
    loadPDFList();
    
    // Update sidebar toggle button position on resize
    window.addEventListener('resize', () => {
        const sidebar = document.querySelector('.sidebar');
        const toggleBtn = document.getElementById('sidebarToggle');
        if (toggleBtn && window.innerWidth > 768) {
            if (sidebar.classList.contains('hidden')) {
                toggleBtn.style.left = '16px';
            } else {
                toggleBtn.style.left = '276px';
            }
        }
    });
    
    // Auto-resize textarea
    messageInput.addEventListener('input', () => {
        messageInput.style.height = 'auto';
        messageInput.style.height = messageInput.scrollHeight + 'px';
        charCount.textContent = messageInput.value.length;
    });
    
    // Send on Enter (Shift+Enter for new line)
    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
});

/**
 * Check system health status from backend.
 */
async function loadSystemStatus() {
    try {
        const response = await fetch('/health');
        const data = await response.json();
        
        if (data.rag_ready) {
            systemStatus.querySelector('span').textContent = 'RAG Sistemi Hazır'; // RAG System Ready
            systemStatus.querySelector('.status-dot').style.background = '#10a37f';
        } else {
            systemStatus.querySelector('span').textContent = 'Yükleniyor...'; // Loading...
            systemStatus.querySelector('.status-dot').style.background = '#f59e0b';
            setTimeout(loadSystemStatus, 3000);
        }
    } catch (error) {
        systemStatus.querySelector('span').textContent = 'Bağlantı Hatası'; // Connection Error
        systemStatus.querySelector('.status-dot').style.background = '#ef4444';
    }
}

/**
 * Load list of indexed PDF files.
 */
async function loadPDFList() {
    try {
        const response = await fetch('/api/pdfs');
        const data = await response.json();
        
        if (data.pdfs && data.pdfs.length > 0) {
            pdfList.innerHTML = data.pdfs.map(pdf => `
                <div class="pdf-item">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                        <polyline points="14 2 14 8 20 8"/>
                    </svg>
                    ${pdf}
                </div>
            `).join('');
            
            chunkCount.textContent = data.total_chunks || '-';
        } else {
            pdfList.innerHTML = '<div class="loading-pdfs">PDF bulunamadı</div>'; // No PDF found
        }
    } catch (error) {
        console.error('Failed to load PDF list:', error);
        pdfList.innerHTML = '<div class="loading-pdfs">Yükleme hatası</div>'; // Load error
    }
}

/**
 * Handle sending user message.
 */
async function sendMessage() {
    const message = messageInput.value.trim();
    
    if (!message || isWaitingForResponse) return;
    
    // Hide welcome screen if visible
    if (welcomeScreen.style.display !== 'none') {
        welcomeScreen.style.display = 'none';
        messagesContainer.style.display = 'block';
    }
    
    // Add user message to UI
    addMessage(message, 'user');
    messageHistory.push({ role: 'user', content: message });
    
    // Clear and reset input
    messageInput.value = '';
    messageInput.style.height = 'auto';
    charCount.textContent = '0';
    
    // Disable input while waiting
    isWaitingForResponse = true;
    sendButton.disabled = true;
    messageInput.disabled = true;
    
    // Show typing indicator
    const typingId = showTypingIndicator();
    
    try {
        // Send request to backend
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: message })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Server error');
        }
        
        // Remove typing indicator
        removeTypingIndicator(typingId);
        
        // Parse response
        const data = await response.json();
        
        // Add bot message to UI
        addMessage(
            data.response || data.answer || 'Cevap bulunamadı', // No answer found
            'bot',
            data.sources || [],
            data.confidence || null,
            data.has_sources || false,
            false, // generated flag
            data.low_confidence || false,
            data.warning || null
        );
        
        // Re-enable input
        isWaitingForResponse = false;
        sendButton.disabled = false;
        messageInput.disabled = false;
        messageInput.focus();
        
    } catch (error) {
        removeTypingIndicator(typingId);
        addMessage(
            'Bağlantı hatası. Lütfen tekrar deneyin.', // Connection error
            'bot',
            [],
            0,
            false
        );
        console.error('Chat error:', error);
        
        // Re-enable input on error
        isWaitingForResponse = false;
        sendButton.disabled = false;
        messageInput.disabled = false;
        messageInput.focus();
    }
}

/**
 * Add a message bubble to the chat interface.
 * @param {string} content - Message text
 * @param {string} role - 'user' or 'bot'
 * @param {Array} sources - List of source objects
 * @param {number} confidence - Confidence score
 */
function addMessage(content, role, sources = [], confidence = null, hasSources = false, generated = false, lowConfidence = false, warning = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message';
    
    // Create Header
    const headerDiv = document.createElement('div');
    headerDiv.className = 'message-header';
    
    const avatar = document.createElement('div');
    avatar.className = `message-avatar ${role}-avatar`;
    avatar.innerHTML = role === 'user' ? Icons.user : Icons.scales;
    
    const name = document.createElement('div');
    name.className = 'message-name';
    name.textContent = role === 'user' ? 'Siz' : 'Hukuki AI';
    
    if (role === 'bot' && generated) {
        const llmBadge = document.createElement('span');
        llmBadge.className = 'llm-badge';
        llmBadge.innerHTML = `${Icons.chip} Llama-3`;
        name.appendChild(llmBadge);
    }
    
    headerDiv.appendChild(avatar);
    headerDiv.appendChild(name);
    
    // Create Content
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    if (role === 'bot') {
        contentDiv.innerHTML = renderMarkdown(content);
    } else {
        contentDiv.textContent = content;
    }
    
    messageDiv.appendChild(headerDiv);
    messageDiv.appendChild(contentDiv);
    
    // Add Metadata (Confidence & Sources) for Bot
    if (role === 'bot' && confidence !== null) {
        const confDiv = document.createElement('div');
        confDiv.className = 'streaming-metadata';
        
        const confClass = confidence >= 70 ? 'high' : confidence >= 50 ? 'medium' : 'low';
        let confHTML = `<div class="confidence-badge confidence-${confClass}">
            <span class="confidence-icon">${confidence >= 70 ? Icons.check : confidence >= 50 ? Icons.alert : Icons.x}</span>
            <span class="confidence-text">Güven: ${Math.round(confidence)}%</span>
        </div>`;
        
        if (lowConfidence && warning) {
            confHTML += `<div class="low-confidence-warning">
                <span class="confidence-icon">${Icons.alert}</span> ${warning}
            </div>`;
        }
        
        confDiv.innerHTML = confHTML;
        messageDiv.appendChild(confDiv);
    }
    
    if (role === 'bot' && sources && sources.length > 0) {
        const sourcesDiv = document.createElement('div');
        sourcesDiv.className = 'message-sources';
        
        sourcesDiv.innerHTML = `
            <div class="sources-title">
                ${Icons.book} ${sources.length} Kaynak Bulundu
            </div>
            ${sources.slice(0, 3).map((source, idx) => {
                const scorePercent = Math.round((source.score || source.similarity_score || 0) * 100);
                const scoreClass = scorePercent >= 70 ? 'high' : scorePercent >= 50 ? 'medium' : 'low';
                return `
                <div class="source-item">
                    <div class="source-header">
                        <div class="source-name">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                                <polyline points="14 2 14 8 20 8"/>
                            </svg>
                            ${source.source_file || source.source}
                            ${source.article ? `<span class="article-badge">📜 ${source.article}</span>` : ''}
                            ${source.page || source.page_number ? `<span class="page-badge">📄 Sayfa ${source.page || source.page_number}</span>` : ''}
                        </div>
                        <span class="source-score confidence-badge confidence-${scoreClass}">${scorePercent}%</span>
                    </div>
                    <div class="source-preview">${source.preview || ''}</div>
                </div>
            `;
            }).join('')}
        `;
        
        contentDiv.appendChild(sourcesDiv);
    }
    
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

/**
 * Display the typing animation.
 * @returns {string} ID of the typing element
 */
function showTypingIndicator() {
    const typingDiv = document.createElement('div');
    const id = 'typing-' + Date.now();
    typingDiv.id = id;
    typingDiv.className = 'message';
    
    typingDiv.innerHTML = `
        <div class="message-header">
            <div class="message-avatar bot-avatar">${Icons.scales}</div>
            <div class="message-name">Hukuki AI</div>
        </div>
        <div class="typing-indicator">
            <span>Dokümanlarda aranıyor</span>
            <div class="typing-dots">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>
    `;
    
    messagesContainer.appendChild(typingDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    
    return id;
}

/**
 * Remove the typing animation.
 * @param {string} id - Element ID
 */
function removeTypingIndicator(id) {
    const typing = document.getElementById(id);
    if (typing) typing.remove();
}

// Quick Questions List (Populated via User Request)
// These are randomized suggestions for the user.
const quickQuestions = [
    'Türkiye Devletinin yönetim şekli nedir?',
    'Türkiye Devletinin başkenti neresidir ve milli marşı nedir?',
    'Yasama yetkisi kime aittir ve bu yetki devredilebilir mi?',
    'Egemenlik kime aittir?',
    'Herkesin kanun önünde eşit olması ilkesi ne anlama gelir?',
    'Milletvekili seçilebilmek için kaç yaşını doldurmuş olmak gerekir?',
    'Cumhurbaşkanı seçilen bir kişinin görev süresi kaç yıldır?',
    'Orman köylüsünün korunması nasıl olur?',
    'Anayasa Mahkemesi kaç üyeden kurulur?',
    'Yüksek mahkemeler hangileridir?',
    'Siyasi partiler hangi tür faaliyetlere girişemezler?',
    'Temel hak ve hürriyetler hangi durumlarda kısmen veya tamamen durdurulabilir?',
    'Hâkimler ve Savcılar Kurulu kaç üyeden oluşur ve başkanı kimdir?',
    'Yargıtay Cumhuriyet Başsavcısı kim tarafından ve kaç yıl için seçilir?',
    'TBMM Genel Kurulu, resmi tatile rastlamadığı takdirde haftanın hangi günleri toplanır?',
    'Bir milletvekili bir yasama yılı içinde izinsiz veya özürsüz olarak toplam 45 birleşimden fazla yok sayılırsa ne olur?',
    'TBMMde kapalı oturum yapılmasını kimler isteyebilir?',
    'Cumhurbaşkanı bütçe kanun teklifini mali yılbaşından ne kadar süre önce Meclise sunmalıdır?',
    'Cumhurbaşkanlığı kararnameleri ile kanunlar arasında farklı hükümler bulunması halinde hangisi uygulanır?',
    'Olağanüstü hal (OHAL) ilanı kararını kim verir ve bu kararın süresi en fazla ne kadar olabilir?',
    'Herkesin özel hayatına ve aile hayatına saygı gösterilmesini isteme hakkı hangi maddede düzenlenmiştir?',
    'Basın hürriyeti kapsamında basımevi kurmak için izin almak ve mali teminat yatırmak şart mıdır?',
    'Milletvekilleri göreve başlarken nerede ve nasıl ant içerler?',
    'Savaş hali ilanına ve Türk Silahlı Kuvvetlerinin yabancı ülkelere gönderilmesine kim izin verir?',
    'Milli Güvenlik Kurulunun gündemi kim tarafından düzenlenir?'
];

// Example Question Handlers
function sendExample(button) {
    const text = button.querySelector('.example-text').textContent;
    messageInput.value = text;
    sendMessage();
}

function sendQuickQuestion(question) {
    messageInput.value = question;
    sendMessage();
}

/**
 * Pick a random question from the list and fill the input.
 * Does NOT send automatically, allowing user modification.
 */
function sendRandomQuestion() {
    if (quickQuestions.length === 0) return;
    const randomIndex = Math.floor(Math.random() * quickQuestions.length);
    const randomQuestion = quickQuestions[randomIndex];
    
    // Set input value
    messageInput.value = randomQuestion;
    
    // Resize textarea to fit content
    messageInput.style.height = 'auto';
    messageInput.style.height = messageInput.scrollHeight + 'px';
    
    // Update char count
    charCount.textContent = messageInput.value.length;
    
    // Focus input
    messageInput.focus();
}

// UI Helpers
function newChat() {
    messagesContainer.innerHTML = '';
    messageHistory = [];
    welcomeScreen.style.display = 'flex';
    messagesContainer.style.display = 'none';
}

function toggleSidebar() {
    const sidebar = document.querySelector('.sidebar');
    const toggleBtn = document.getElementById('sidebarToggle');
    const isMobile = window.innerWidth <= 768;
    
    if (isMobile) {
        sidebar.classList.toggle('active');
    } else {
        sidebar.classList.toggle('hidden');
        if (toggleBtn) {
            if (sidebar.classList.contains('hidden')) {
                toggleBtn.style.left = '16px';
            } else {
                toggleBtn.style.left = '276px';
            }
        }
    }
}

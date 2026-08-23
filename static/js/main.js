document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const docList = document.getElementById("doc-list");
    const thresholdSlider = document.getElementById("threshold-slider");
    const thresholdVal = document.getElementById("threshold-val");
    const topkSlider = document.getElementById("topk-slider");
    const topkVal = document.getElementById("topk-val");

    const welcomeScreen = document.getElementById("welcome-screen");
    const messageContainer = document.getElementById("message-container");
    const chatInput = document.getElementById("chat-input");
    const sendBtn = document.getElementById("send-btn");
    const statusMode = document.getElementById("status-mode");
    const uploadProgressContainer = document.getElementById("upload-progress-container");
    const uploadProgressBar = document.getElementById("upload-progress-bar");

    // Multi-Chat DOM Elements
    const chatListContainer = document.getElementById("chat-list");
    const btnNewChat = document.getElementById("btn-new-chat");

    // Multi-Chat State
    let currentChatId = null;
    let hasDocuments = false;

    // Load initial chats on startup
    loadChats();

    // Slider Listeners
    thresholdSlider.addEventListener("input", (e) => {
        thresholdVal.textContent = parseFloat(e.target.value).toFixed(2);
    });
    topkSlider.addEventListener("input", (e) => {
        topkVal.textContent = e.target.value;
    });

    // Auto-growing Textarea Input & Send Activation
    chatInput.addEventListener("input", () => {
        chatInput.style.height = "auto";
        chatInput.style.height = (chatInput.scrollHeight - 16) + "px";
        sendBtn.disabled = chatInput.value.trim() === "";
    });

    // Keydown Listener (Enter submits, Shift+Enter adds newline)
    chatInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            if (chatInput.value.trim() !== "") {
                sendMessage();
            }
        }
    });

    sendBtn.addEventListener("click", () => {
        if (chatInput.value.trim() !== "") {
            sendMessage();
        }
    });

    // Starter Cards Click
    document.querySelectorAll(".starter-card").forEach(card => {
        card.addEventListener("click", () => {
            const query = card.getAttribute("data-query");
            chatInput.value = query;
            sendBtn.disabled = false;
            sendMessage();
        });
    });


    // Drag and Drop Event Listeners
    dropZone.addEventListener("click", () => fileInput.click());

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            uploadFiles(e.target.files);
        }
    });

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("dragover");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            uploadFiles(e.dataTransfer.files);
        }
    });

    // New Chat Button Click
    btnNewChat.addEventListener("click", () => {
        createNewChat();
    });

    // Load Chat List
    async function loadChats() {
        try {
            const response = await fetch("/chats");
            const data = await response.json();
            const chats = data.chats || [];

            chatListContainer.innerHTML = "";
            if (chats.length === 0) {
                // If no chats exist, auto-create one default chat
                await createNewChat();
                return;
            }

            chats.forEach(chat => {
                const chatItem = document.createElement("div");
                chatItem.className = `chat-item ${chat.chat_id === currentChatId ? 'active' : ''}`;
                chatItem.setAttribute("data-chat-id", chat.chat_id);
                chatItem.addEventListener("click", () => selectChat(chat.chat_id));

                chatItem.innerHTML = `
                    <span class="chat-item-icon">💬</span>
                    <span class="chat-item-name" title="${chat.name}">${chat.name}</span>
                    <button class="btn-delete-chat" title="Delete Chat">🗑️</button>
                `;

                // Attach delete action
                const delBtn = chatItem.querySelector(".btn-delete-chat");
                delBtn.addEventListener("click", (e) => {
                    e.stopPropagation();
                    deleteChat(chat.chat_id);
                });

                chatListContainer.appendChild(chatItem);
            });

            // If currentChatId is not set or not in the list, select the first chat
            const currentChatExists = chats.some(c => c.chat_id === currentChatId);
            if (!currentChatId || !currentChatExists) {
                selectChat(chats[0].chat_id);
            }
        } catch (err) {
            console.error("Failed to load chats: ", err);
        }
    }

    // Create New Chat
    async function createNewChat() {
        try {
            const response = await fetch("/chats", { method: "POST" });
            const result = await response.json();
            if (result.success) {
                currentChatId = result.chat.chat_id;
                await loadChats();
                selectChat(currentChatId);
            }
        } catch (err) {
            console.error("Failed to create new chat: ", err);
        }
    }

    // Delete Chat
    async function deleteChat(chatId) {
        if (!confirm("Are you sure you want to delete this chat workspace and all its files?")) {
            return;
        }
        try {
            const response = await fetch(`/chats/${chatId}`, { method: "DELETE" });
            const result = await response.json();
            if (result.success) {
                if (currentChatId === chatId) {
                    currentChatId = null;
                }
                await loadChats();
            }
        } catch (err) {
            console.error("Failed to delete chat: ", err);
        }
    }

    // Select Chat Session
    async function selectChat(chatId) {
        currentChatId = chatId;

        // Update active class in sidebar items
        document.querySelectorAll(".chat-item").forEach(item => {
            if (item.getAttribute("data-chat-id") === chatId) {
                item.classList.add("active");
            } else {
                item.classList.remove("active");
            }
        });

        // Clear message container
        messageContainer.innerHTML = "";

        // Reload documents and history
        await loadDocuments();
        await loadChatHistory(chatId);
    }

    // Fetch Documents for Current Chat & Render
    async function loadDocuments() {
        if (!currentChatId) return;
        try {
            const response = await fetch(`/documents?chat_id=${currentChatId}`);
            const data = await response.json();
            const docs = data.documents || [];

            if (docs.length === 0) {
                docList.innerHTML = '<div class="empty-docs-placeholder">No documents ingested.</div>';
                statusMode.textContent = "Standard Chatbot Mode";
                hasDocuments = false;
            } else {
                hasDocuments = true;
                statusMode.textContent = "Document-Grounded RAG Mode";
                docList.innerHTML = "";
                docs.forEach(doc => {
                    const ext = doc.name.split('.').pop().toLowerCase();
                    let icon = "📄";
                    if (["png", "jpg", "jpeg"].includes(ext)) icon = "🖼️";
                    if (ext === "pdf") icon = "📕";
                    if (["doc", "docx"].includes(ext)) icon = "📘";
                    if (ext === "txt") icon = "📝";

                    const docItem = document.createElement("div");
                    docItem.className = "doc-item";
                    docItem.innerHTML = `
                        <span class="doc-item-icon">${icon}</span>
                        <div class="doc-item-info">
                            <div class="doc-item-name" title="${doc.name}">${doc.name}</div>
                            <div class="doc-item-size">${doc.size_kb} KB</div>
                        </div>
                    `;
                    docList.appendChild(docItem);
                });
            }
        } catch (err) {
            console.error("Failed loading documents: ", err);
        }
    }

    // Load Chat History and Render bubbles
    async function loadChatHistory(chatId) {
        try {
            const response = await fetch(`/chats/${chatId}/history`);
            const data = await response.json();
            const history = data.history || [];

            if (history.length === 0) {
                welcomeScreen.style.display = "flex";
                messageContainer.style.display = "none";
            } else {
                welcomeScreen.style.display = "none";
                messageContainer.style.display = "flex";

                history.forEach(msg => {
                    const sender = msg.role === "user" ? "user" : "bot";

                    let contentHtml = (msg.content || "").replace(/\n/g, "<br>");

                    let sourcesHtml = "";

                    // Restore persisted citations
                    if (
                        sender === "bot" &&
                        msg.sources &&
                        msg.sources.length > 0
                    ) {
                        sourcesHtml = `
                            <div class="sources-container">
                                <div class="sources-title">Cited Sources</div>
                                ${msg.sources.map(src => `
                                    <span
                                        class="source-badge"
                                        title="${src.source_path || ""}"
                                    >
                                        [${src.citation_id}] ${src.file_name}
                                    </span>
                                `).join("")}
                            </div>
                        `;
                    }

                    appendMessageBubble(
                        sender,
                        `
                        <div class="message-content">
                            <p>${contentHtml}</p>
                            ${sourcesHtml}
                        </div>
                        `
                    );
                });

                scrollToBottom();
            }
        } catch (err) {
            console.error("Failed loading chat history: ", err);
        }
    }

    // Upload Files Asynchronously
    async function uploadFiles(files) {
        if (!currentChatId) {
            alert("No active chat session. Please create one.");
            return;
        }

        const formData = new FormData();
        formData.append("chat_id", currentChatId);
        for (let i = 0; i < files.length; i++) {
            formData.append("files", files[i]);
        }

        uploadProgressContainer.style.display = "block";
        uploadProgressBar.style.width = "40%";

        try {
            const response = await fetch("/upload", {
                method: "POST",
                body: formData
            });
            uploadProgressBar.style.width = "80%";

            const result = await response.json();
            uploadProgressBar.style.width = "100%";

            setTimeout(() => {
                uploadProgressContainer.style.display = "none";
                uploadProgressBar.style.width = "0%";
            }, 500);

            if (result.success) {
                let msg = `Successfully uploaded and ingested ${result.ingested_count} document(s).`;
                if (result.errors && result.errors.length > 0) {
                    msg += "\n\nIssues encountered:\n" + result.errors.join("\n");
                }
                alert(msg);
                loadDocuments();
            } else {
                alert("Upload failed: " + result.error);
            }
        } catch (err) {
            uploadProgressContainer.style.display = "none";
            alert("Upload connection failed: " + err);
        }
    }

    // Send Message Implementation
    async function sendMessage() {
        const query = chatInput.value.trim();
        if (!query || !currentChatId) return;

        // Clear input field
        chatInput.value = "";
        chatInput.style.height = "auto";
        sendBtn.disabled = true;

        // Switch layouts if it's the first message
        if (welcomeScreen.style.display !== "none") {
            welcomeScreen.style.display = "none";
            messageContainer.style.display = "flex";
        }

        // 1. Append User Message
        appendMessageBubble("user", `<div class="message-content"><p>${query.replace(/\n/g, "<br>")}</p></div>`);

        // 2. Append Bot Placeholder / Loader
        const botBubbleId = "bot-bubble-" + Date.now();
        appendMessageBubble("bot", `
            <div class="thinking-bubble">
                <span class="dot"></span>
                <span class="dot"></span>
                <span class="dot"></span>
            </div>
        `, botBubbleId);

        scrollToBottom();

        // 3. Post Query to Backend
        try {
            const payload = {
                query: query,
                score_threshold: parseFloat(thresholdSlider.value),
                top_k: parseInt(topkSlider.value),
                chat_id: currentChatId
            };

            const response = await fetch("/query", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                throw new Error("HTTP returned status code " + response.status);
            }

            const data = await response.json();

            // Format answer with line breaks
            let answerHtml = data.answer ? data.answer.replace(/\n/g, "<br>") : "";

            // Append sources list if present
            let sourcesHtml = "";
            if (data.sources && data.sources.length > 0) {
                sourcesHtml += '<div class="sources-container"><div class="sources-title">Cited Sources</div>';
                data.sources.forEach(src => {
                    sourcesHtml += `<span class="source-badge" title="${src.source_path}">[${src.citation_id}] ${src.file_name}</span>`;
                });
                sourcesHtml += '</div>';
            }

            // Append diagnostics if available
            let diagnosticsHtml = "";
            if (data.diagnostic_logs) {
                const logsString = JSON.stringify(data.diagnostic_logs, null, 2);
                diagnosticsHtml += `
                    <span class="diagnostic-toggle" onclick="toggleDiagnostics('${botBubbleId}')">⚙️ View diagnostics logs</span>
                    <pre class="diagnostic-box" id="diag-${botBubbleId}">${logsString}</pre>
                `;
            }

            // Update placeholder bubble with response content
            const bubbleEl = document.getElementById(botBubbleId);
            bubbleEl.innerHTML = `
                <div class="message-content">
                    <p>${answerHtml}</p>
                    ${sourcesHtml}
                    ${diagnosticsHtml}
                </div>
            `;

            // Auto rename title if it was the first user message
            const activeChatItem = document.querySelector(".chat-item.active");
            if (activeChatItem) {
                const nameEl = activeChatItem.querySelector(".chat-item-name");
                if (nameEl && nameEl.textContent === "New Chat") {
                    setTimeout(() => {
                        fetch("/chats")
                            .then(res => res.json())
                            .then(cData => {
                                const matched = (cData.chats || []).find(c => c.chat_id === currentChatId);
                                if (matched) {
                                    nameEl.textContent = matched.name;
                                    nameEl.title = matched.name;
                                }
                            });
                    }, 500);
                }
            }

        } catch (err) {
            const bubbleEl = document.getElementById(botBubbleId);
            bubbleEl.innerHTML = `
                <div class="message-content text-danger" style="color: var(--danger)">
                    <p>⚠️ Error: Could not connect to local generation provider. Ensure Ollama is running.</p>
                    <small style="font-size: 11px; color: var(--text-muted)">Details: ${err.message}</small>
                </div>
            `;
        }

        scrollToBottom();
    }

    // Helper: Add bubble to log UI
    function appendMessageBubble(sender, content, id = null) {
        const bubbleRow = document.createElement("div");
        bubbleRow.className = `message-row ${sender}`;

        const bubble = document.createElement("div");
        bubble.className = "chat-bubble";
        if (id) bubble.id = id;
        bubble.innerHTML = content;

        bubbleRow.appendChild(bubble);
        messageContainer.appendChild(bubbleRow);
    }

    // Helper: Scroll Message Pane
    function scrollToBottom() {
        messageContainer.scrollTop = messageContainer.scrollHeight;
    }
});

// Toggle helper for Diagnostics View (attached to window since it is called inline)
window.toggleDiagnostics = (bubbleId) => {
    const diagBox = document.getElementById(`diag-${bubbleId}`);
    if (diagBox) {
        const isVisible = diagBox.style.display === "block";
        diagBox.style.display = isVisible ? "none" : "block";
        const messageContainer = document.getElementById("message-container");
        messageContainer.scrollTop = messageContainer.scrollHeight;
    }
};

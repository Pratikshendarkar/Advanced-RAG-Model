from dotenv import load_dotenv

css = '''
<style>
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    padding: 1rem 0;
}

.chat-message {
    display: flex;
    gap: 0.75rem;
    margin-bottom: 1.5rem;
    animation: fadeIn 0.4s ease-in;
    align-items: flex-start;
}

@keyframes fadeIn {
    from { 
        opacity: 0; 
        transform: translateY(10px); 
    }
    to { 
        opacity: 1; 
        transform: translateY(0); 
    }
}

.chat-message.user {
    flex-direction: row-reverse;
    justify-content: flex-start;
}

.chat-message.bot {
    flex-direction: row;
    justify-content: flex-start;
}

.chat-message .avatar {
    flex-shrink: 0;
    width: 36px;
    height: 36px;
}

.chat-message .avatar img {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    object-fit: cover;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
}

.chat-message .message {
    max-width: 70%;
    padding: 0.875rem 1.125rem;
    border-radius: 18px;
    line-height: 1.6;
    word-wrap: break-word;
    font-size: 0.95rem;
}

.chat-message.user .message {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-bottom-right-radius: 4px;
    box-shadow: 0 2px 8px rgba(102, 126, 234, 0.4);
}

.chat-message.bot .message {
    background: #f7f7f8;
    color: #1f1f1f;
    border-bottom-left-radius: 4px;
    border: 1px solid #e5e5e5;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}

/* Better text styling */
.chat-message.user .message p {
    margin: 0;
    color: white;
}

.chat-message.bot .message p {
    margin: 0;
    color: #1f1f1f;
}

/* Code blocks in bot messages */
.chat-message.bot .message pre {
    background: #2d2d2d;
    color: #f8f8f2;
    padding: 0.75rem 1rem;
    border-radius: 8px;
    overflow-x: auto;
    margin: 0.75rem 0;
    font-size: 0.85rem;
    line-height: 1.4;
}

.chat-message.bot .message code {
    background: #e8e8e8;
    color: #d63384;
    padding: 0.2rem 0.4rem;
    border-radius: 4px;
    font-family: 'Courier New', monospace;
    font-size: 0.9rem;
}

.chat-message.bot .message pre code {
    background: transparent;
    color: #f8f8f2;
    padding: 0;
}

/* Lists in messages */
.chat-message .message ul,
.chat-message .message ol {
    margin: 0.5rem 0;
    padding-left: 1.5rem;
}

.chat-message .message li {
    margin: 0.25rem 0;
}

/* Links in messages */
.chat-message .message a {
    color: #667eea;
    text-decoration: underline;
}

.chat-message.user .message a {
    color: white;
    text-decoration: underline;
}

/* Bold text */
.chat-message .message strong {
    font-weight: 600;
}

/* Italic text */
.chat-message .message em {
    font-style: italic;
}

/* Blockquotes */
.chat-message.bot .message blockquote {
    border-left: 3px solid #667eea;
    padding-left: 1rem;
    margin: 0.75rem 0;
    color: #555;
    font-style: italic;
}

/* Horizontal rules */
.chat-message.bot .message hr {
    border: none;
    border-top: 1px solid #e5e5e5;
    margin: 1rem 0;
}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: #c1c1c1;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: #a8a8a8;
}

/* Responsive design */
@media (max-width: 768px) {
    .chat-message .message {
        max-width: 85%;
        font-size: 0.9rem;
    }
    
    .chat-message .avatar {
        width: 32px;
        height: 32px;
    }
    
    .chat-message .avatar img {
        width: 32px;
        height: 32px;
    }
}

/* Typing indicator animation (optional) */
@keyframes typing {
    0%, 100% { opacity: 0.3; }
    50% { opacity: 1; }
}

.typing-indicator {
    display: flex;
    gap: 4px;
    padding: 0.5rem;
}

.typing-indicator span {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #999;
    animation: typing 1.4s infinite;
}

.typing-indicator span:nth-child(2) {
    animation-delay: 0.2s;
}

.typing-indicator span:nth-child(3) {
    animation-delay: 0.4s;
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.ibb.co/cN0nmSj/Screenshot-2023-05-28-at-02-37-21.png" alt="AI">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://cdn-icons-png.flaticon.com/512/6596/6596121.png" alt="User">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''
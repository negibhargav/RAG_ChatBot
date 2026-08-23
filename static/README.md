# Static UI Assets

This directory contains the frontend client assets for the RAG Bot web application.

---

## 📁 File Structure

- `css/style.css`: Modern dark theme design system using CSS variables, custom scrollbars, glassmorphism card styling, responsive flex containers, and text contrast rules.
- `js/main.js`: Vanilla JavaScript controller managing UI state:
  - Multi-chat workspace sidebar rendering and chat switching.
  - File drag-and-drop & progress bar animation.
  - Auto-growing text input height calculation.
  - Asynchronous fetch API calls (`/chats`, `/documents`, `/upload`, `/query`).
  - Chat history re-rendering with citation badges and collapsible diagnostics logs.

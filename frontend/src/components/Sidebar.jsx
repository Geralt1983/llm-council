import { useState } from 'react';
import './Sidebar.css';

export default function Sidebar({
  conversations,
  currentConversationId,
  onSelectConversation,
  onNewConversation,
  onDeleteConversation,
  onExportConversation,
  onOpenSettings,
  onOpenAnalytics,
}) {
  const [menuOpen, setMenuOpen] = useState(null);

  const handleMenuToggle = (e, convId) => {
    e.stopPropagation();
    setMenuOpen(menuOpen === convId ? null : convId);
  };

  const handleDelete = (e, convId) => {
    e.stopPropagation();
    if (window.confirm('Are you sure you want to delete this conversation?')) {
      onDeleteConversation(convId);
    }
    setMenuOpen(null);
  };

  const handleExport = (e, convId, format) => {
    e.stopPropagation();
    onExportConversation(convId, format);
    setMenuOpen(null);
  };

  return (
    <div className="sidebar">
      <div className="sidebar-header">
        <h1>LLM Council</h1>
        <button className="new-conversation-btn" onClick={onNewConversation}>
          + New
        </button>
      </div>

      <div className="conversation-list">
        {conversations.length === 0 ? (
          <div className="no-conversations">No conversations yet</div>
        ) : (
          conversations.map((conv) => (
            <div
              key={conv.id}
              className={`conversation-item ${
                conv.id === currentConversationId ? 'active' : ''
              }`}
              onClick={() => onSelectConversation(conv.id)}
            >
              <div className="conversation-content">
                <div className="conversation-title">
                  {conv.title || 'New Conversation'}
                </div>
                <div className="conversation-meta">
                  {conv.message_count} messages
                </div>
              </div>
              <div className="conversation-actions">
                <button
                  className="menu-btn"
                  onClick={(e) => handleMenuToggle(e, conv.id)}
                  title="More options"
                >
                  ...
                </button>
                {menuOpen === conv.id && (
                  <div className="conversation-menu">
                    <button onClick={(e) => handleExport(e, conv.id, 'markdown')}>
                      Export Markdown
                    </button>
                    <button onClick={(e) => handleExport(e, conv.id, 'json')}>
                      Export JSON
                    </button>
                    <button
                      className="delete-btn"
                      onClick={(e) => handleDelete(e, conv.id)}
                    >
                      Delete
                    </button>
                  </div>
                )}
              </div>
            </div>
          ))
        )}
      </div>

      <div className="sidebar-footer">
        <button className="footer-btn" onClick={onOpenAnalytics}>
          <span className="footer-icon">&#128202;</span>
          Analytics
        </button>
        <button className="footer-btn" onClick={onOpenSettings}>
          <span className="footer-icon">&#9881;</span>
          Settings
        </button>
      </div>
    </div>
  );
}

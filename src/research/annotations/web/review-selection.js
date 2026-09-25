export const capturePassageSelection = (passage, task) => {
  if (!passage?.isConnected || passage.textContent !== task.text
      || (passage.dataset.taskId && passage.dataset.taskId !== task.task_id)
      || (passage.dataset.textRevision && passage.dataset.textRevision !== task.text_revision)) return null;
  const selection = window.getSelection();
  if (!selection?.rangeCount || selection.isCollapsed) return null;
  const range = selection.getRangeAt(0);
  if (!passage.contains(range.startContainer) || !passage.contains(range.endContainer)) return null;
  const prefix = document.createRange();
  prefix.selectNodeContents(passage);
  prefix.setEnd(range.startContainer, range.startOffset);
  const text = range.toString();
  if (!text.trim()) return null;
  const localStart = Array.from(prefix.toString()).length;
  const length = Array.from(text).length;
  if (Array.from(task.text).slice(localStart, localStart + length).join('') !== text) return null;
  const start = task.offset_base + localStart;
  const end = start + length;
  const box = range.getBoundingClientRect();
  return {taskId: task.task_id, textRevision: task.text_revision, start, end, text,
    rect: {left: box.left, right: box.right, top: box.top, bottom: box.bottom}};
};

export const positionPopover = (rect, viewport, menuSize) => {
  const margin = 8;
  const maxLeft = Math.max(0, viewport.width - menuSize.width - margin);
  const maxTop = Math.max(0, viewport.height - menuSize.height - margin);
  const below = rect.bottom + margin;
  const top = below + menuSize.height <= viewport.height - margin ? below : rect.top - menuSize.height - margin;
  return {left: Math.max(0, Math.min(Math.max(margin, rect.left), maxLeft)),
    top: Math.max(0, Math.min(top, maxTop))};
};

export function highlightText(text: string, query: string): string {
  if (!query || query.length < 3) {
    return escapeHtml(text);
  }

  const queryWords = query
    .toLowerCase()
    .split(/\s+/)
    .filter((word) => word.length >= 2);

  if (queryWords.length === 0) {
    return escapeHtml(text);
  }

  // Create regex pattern to match any of the query words (case-insensitive)
  const pattern = new RegExp(
    `(${queryWords.map((word) => escapeRegex(word)).join('|')})`,
    'gi'
  );

  const escaped = escapeHtml(text);
  return escaped.replace(pattern, '<mark class="bg-yellow-200 dark:bg-yellow-900">$1</mark>');
}

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function escapeRegex(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}


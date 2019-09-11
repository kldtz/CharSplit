"""RFC 4647 language tag matching."""

def canonicalize(tag: str, default: str) -> str:
    """Lowercase tag, and if it is None or empty, replace it with default."""
    if not tag:
        tag = default
    return tag.lower().replace('_', '-')

def language_basic_filter_match(needed: str, available: str, default: str = 'en') -> bool:
    """Performs a filter match according to RFC 4647.
       See <https://tools.ietf.org/html/rfc4647#section-3.4> for details.
       The 'needed' argument is the language we need,
       and corresponds to the 'language range' in the RFC.
       The 'available' argument is what we have,
       and corresponds to the 'language tag' in the RFC.
       If either argument is None or empty, we assume the default.
    """
    # Canonicalize inputs
    needed = canonicalize(needed, default)
    available = canonicalize(available, default)

    # Perform basic matching
    if needed == '*':
        # any available tag will do
        return True
    if needed == available:
        # exact match
        return True
    # accept a narrower (longer) available tag
    return available.startswith(needed + '-')

def language_lookup_match(needed: str, available: str, default: str = 'en') -> bool:
    """Performs a lookup match according to RFC 4647.
       See <https://tools.ietf.org/html/rfc4647#section-3.4> for details.
       The 'needed' argument is the language we need,
       and corresponds to the 'language range' in the RFC.
       The 'available' argument is what we have,
       and corresponds to the 'language tag' in the RFC.
       If either argument is None or empty, we assume the default.
    """

    # Split canonicalized arguments
    needed_subtags = canonicalize(needed, default).split('-')
    available_subtags = canonicalize(available, default).split('-')

    # Match plain "*" tag, but not any other "*" subtags
    if "*" in needed_subtags:
        return len(needed_subtags) == 1

    # Try to match. If we fail, reduce 'available' by one step
    # (going from more specific to less specific) and try again.
    while needed_subtags:
        if needed_subtags == available_subtags:
            return True
        needed_subtags = needed_subtags[:-1]
        # Remove a single-letter tag if present
        try:
            if len(needed_subtags[-1]) == 1:
                needed_subtags = available_subtags[:-1]
        except IndexError:
            pass
    return False

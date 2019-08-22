"""RFC 4647 language tag matching."""


def language_lookup_match(needed: str, available: str, default: str = 'en') -> bool:
    """Performs a lookup match according to RFC 4647.
       See <https://tools.ietf.org/html/rfc4647#section-3.4> for details.
       The 'needed' argument is the language we need,
       and corresponds to the 'language range' in the RFC.
       The 'available' argument is what we have,
       and corresponds to the 'language tag' in the RFC.
       If either argument is None or empty, we assume the default.
    """

    # Canonicalize arguments
    if not needed:
        needed = default
    needed = needed.lower().replace('_', '-')
    needed_subtags = needed.split('-')
    if not available:
        available = default
    available = available.lower().replace('_', '-')
    available_subtags = available.split('-')

    # Match plain "*" tag, but not any other "*" subtags
    if "*" in needed:
      return len(needed) == 1

    # Try to match. If we fail, reduce 'needed' by one step
    while needed_subtags:
        if needed_subtags == available_subtags:
             return True
        needed_subtags = needed_subtags[:-1]
        # Remove a single-letter tag if present
        try:
            if len(needed_subtags[-1]) == 1:
              needed_subtags = needed_subtags[:-1]
        except IndexError:
              pass
    return False

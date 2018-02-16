int _assert_internal(int result,
                     __global AssertBuffer* restrict assertBuffer,
                     const __constant char* restrict file,
                     uint line,
                     const __constant char* restrict expr)
{
    if (result)
        return 0;

    if (atomic_inc(&(assertBuffer->details.assertCount)))
        return 1; // There already is a registered assertion failure

    // Fill in al the numerical details
    for (unsigned i = 0; i < 4; ++i)
        assertBuffer->details.globalId[i] = get_global_id(i);
    assertBuffer->details.line = line;

    uint i = sizeof(assertBuffer->details);

    // Copy file name and separator
    while (i < ASSERT_BUFFER_SIZE && *file)
    {
        assertBuffer->text[i] = *file;
        ++i;
        ++file;
    }
    if (i == ASSERT_BUFFER_SIZE)
        return 1;
    assertBuffer->text[i++] = '\0';

    // Copy failed expression and separator
    while (i < ASSERT_BUFFER_SIZE && *expr)
    {
        assertBuffer->text[i] = *expr;
        ++i;
        ++expr;
    }
    if (i == ASSERT_BUFFER_SIZE)
        return 1;
    assertBuffer->text[i++] = '\0';

    return 1;
}

// vim: filetype=c

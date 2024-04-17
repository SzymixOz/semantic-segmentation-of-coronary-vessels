def unpack_mask(mask, shape=(512, 512)):
    """Unpack segmentation mask sent in HTTP request.
    Args:
        mask (bytes): Packed segmentation mask.
    Returns:
        np.array: Numpy array containing segmentation mask.
    """
    mask = base64.b64decode(mask)
    mask = zlib.decompress(mask)
    mask = list(mask)
    mask = np.array(mask, dtype=np.uint8)
    # pylint:disable=too-many-function-args
    mask = mask.reshape(-1, *shape)
    mask = mask.squeeze()
    return mask

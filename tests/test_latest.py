from edge_detector.latest import LatestValueBuffer


def test_publish_replaces_unread_value() -> None:
    buffer: LatestValueBuffer[int] = LatestValueBuffer()

    assert buffer.publish(1) is True
    assert buffer.publish(2) is True

    assert buffer.take() == 2
    assert buffer.stats() == {"published": 2, "replaced": 1}


def test_taken_value_does_not_count_as_replaced() -> None:
    buffer: LatestValueBuffer[int] = LatestValueBuffer()

    buffer.publish(1)
    assert buffer.take() == 1
    buffer.publish(2)

    assert buffer.take() == 2
    assert buffer.stats() == {"published": 2, "replaced": 0}


def test_close_unblocks_empty_buffer_and_rejects_publish() -> None:
    buffer: LatestValueBuffer[int] = LatestValueBuffer()
    buffer.close()

    assert buffer.take() is None
    assert buffer.publish(1) is False

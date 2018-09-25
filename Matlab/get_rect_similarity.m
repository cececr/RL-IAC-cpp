function sim = get_rect_similarity(rect1, rect2)
    rect1 = rect1 + [0 0 rect1(1) rect1(2)];
    rect2 = rect2 + [0 0 rect2(1) rect2(2)];
    minbound = min(rect1, rect2);
    maxbound = max(rect1,rect2);
    sim = ((minbound(3)-maxbound(1)) * (minbound(4)-maxbound(2))) ...
          / ((maxbound(3)-minbound(1)) * (maxbound(4)-minbound(2)));
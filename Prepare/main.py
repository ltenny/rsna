from labelrecord import LabelRecord


def main(argv=None):
    print('loading records')
    lr = LabelRecord()
    label_records = lr.load('../input/stage_1_train_labels.csv')
    print('done reading %d records' % len(label_records))
    cnt = 0
    for (_,v) in label_records.items():
        if v.hasBoundingBox:
            cnt = cnt + 1
    print('%d records have bounding boxes' % cnt)


if __name__ == '__main__':
    main()
    
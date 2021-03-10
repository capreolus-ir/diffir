from collections import defaultdict
import ir_datasets

_logger = ir_datasets.log.easy()


def load_trec_run(run_fn):
    _logger.info("Loading run file: {}".format(run_fn))
    run = defaultdict(dict)

    for seq, line in enumerate(open(run_fn)):
        line = line.strip()
        if not len(line):
            continue

        qid, _, docid, _, score, _ = line.split()
        run[qid][docid] = float(score)

    return run

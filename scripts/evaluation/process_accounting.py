"""Historical broker adapter; current accounting implementation is in storage.py."""
from src.evaluation.roleplay_benchmarks.storage import ProcessLedger, LedgerFailure


def install(full, options):
    """Wrap the already-installed transport without changing its cache identity."""
    previous = full.Broker

    class ProcessAccountingBroker(previous):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.accounting = ProcessLedger(self.journal / 'ledgers', **options)

        def persist_entry(self, entry):
            # The transport calls this under its accounting lock. This method
            # does no filesystem I/O, no flush, and never waits for queue space.
            self.accounting.append(entry)

        def _check_accounting(self):
            try:
                self.accounting.check()
            except LedgerFailure as exc:
                with self.lock:
                    self.halt = 'accounting_writer_failure'
                self.storage_failure(exc, 'asynchronous_accounting')

        def send(self, payload):
            self._check_accounting()
            result = super().send(payload)
            if result[0] != 200:
                # Unlike successes, HTTP failures have no full response journal
                # to rebuild their error metadata from. Confirm their ledger
                # before delivery, after the transport released its global lock.
                try:
                    self.accounting.flush()
                except LedgerFailure:
                    self._check_accounting()
            return result

        def snapshot(self):
            self._check_accounting()
            value = super().snapshot()
            value['accounting_writer'] = self.accounting.snapshot()
            value['current_ledger'] = self.accounting.last_segment
            return value

        def flush_accounting(self):
            self.accounting.flush()

        def close_accounting(self):
            self.accounting.close()

    full.Broker = ProcessAccountingBroker

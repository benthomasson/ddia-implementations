"""Total Order Broadcast protocol built on single-decree Paxos."""


class ConsensusInstance:
    """A single consensus instance (single-decree Paxos) for one slot."""

    def __init__(self, slot, num_nodes):
        self.slot = slot
        self.num_nodes = num_nodes
        self._decided = False
        self._decided_value = None
        # Acceptor state
        self.promised = -1  # highest proposal number promised
        self.accepted_proposal = None
        self.accepted_value = None

    @property
    def is_decided(self):
        return self._decided

    @property
    def decided_value(self):
        return self._decided_value

    def force_decide(self, value):
        """Force a decision (used for catch-up during recovery)."""
        self._decided = True
        self._decided_value = value

    def prepare(self, proposal_number, proposer_id):
        """Phase 1: Handle a prepare request."""
        if proposal_number > self.promised:
            self.promised = proposal_number
            return {
                'promised': True,
                'accepted_proposal': self.accepted_proposal,
                'accepted_value': self.accepted_value,
            }
        return {
            'promised': False,
            'accepted_proposal': self.accepted_proposal,
            'accepted_value': self.accepted_value,
        }

    def accept(self, proposal_number, value, proposer_id):
        """Phase 2: Handle an accept request."""
        if proposal_number >= self.promised:
            self.promised = proposal_number
            self.accepted_proposal = proposal_number
            self.accepted_value = value
            return {'accepted': True}
        return {'accepted': False}


class TOBNode:
    """A total order broadcast node."""

    def __init__(self, node_id, peer_ids):
        self.node_id = node_id
        self.peer_ids = peer_ids
        self.all_ids = sorted([node_id] + list(peer_ids))
        self.num_nodes = len(self.all_ids)
        self.majority = self.num_nodes // 2 + 1

        self._delivered = []  # list of (slot, message)
        self._next_slot = 0  # next slot to deliver
        self._callbacks = []
        self._alive = True

        # Consensus instances per slot
        self._instances = {}

        # Pending messages to broadcast (queue)
        self._pending = []

        # Current proposal state: {slot: {round, phase, promises, accepts, value, highest_accepted}}
        self._proposals = {}

    @property
    def delivered_messages(self):
        return list(self._delivered)

    @property
    def next_slot(self):
        return self._next_slot

    @property
    def is_alive(self):
        return self._alive

    def _get_instance(self, slot):
        if slot not in self._instances:
            self._instances[slot] = ConsensusInstance(slot, self.num_nodes)
        return self._instances[slot]

    def broadcast(self, message):
        """Queue a message for broadcast."""
        self._pending.append(message)

    def on_deliver(self, callback):
        self._callbacks.append(callback)

    def _make_proposal_number(self, round_num):
        """Unique proposal number: round * num_nodes + node_id."""
        return round_num * self.num_nodes + self.node_id

    def _deliver_decided_slots(self):
        """Deliver any consecutively decided slots starting from _next_slot."""
        msgs = []
        while True:
            inst = self._instances.get(self._next_slot)
            if inst and inst.is_decided:
                val = inst.decided_value
                self._delivered.append((self._next_slot, val))
                for cb in self._callbacks:
                    cb(self._next_slot, val)
                self._next_slot += 1
            else:
                break
        return msgs

    def tick(self):
        """Drive protocol forward."""
        if not self._alive:
            return []

        outgoing = []

        # Try to deliver any decided slots
        self._deliver_decided_slots()

        # Start proposals for pending messages
        while self._pending:
            msg = self._pending.pop(0)
            # Find the next undecided slot to propose for
            slot = self._find_proposal_slot()
            self._start_proposal(slot, msg, outgoing)

        # Retry any in-progress proposals that need re-proposing
        for slot in list(self._proposals.keys()):
            prop = self._proposals[slot]
            inst = self._get_instance(slot)
            if inst.is_decided:
                # Slot was decided by someone else - re-propose our value for next slot
                if inst.decided_value != prop['value']:
                    new_slot = self._find_proposal_slot()
                    value = prop['value']
                    del self._proposals[slot]
                    self._start_proposal(new_slot, value, outgoing)
                else:
                    del self._proposals[slot]

        return outgoing

    def _find_proposal_slot(self):
        """Find the lowest undecided slot not already being proposed for."""
        slot = self._next_slot
        while True:
            inst = self._instances.get(slot)
            if inst and inst.is_decided:
                slot += 1
                continue
            if slot in self._proposals:
                slot += 1
                continue
            return slot

    def _start_proposal(self, slot, value, outgoing):
        """Start Paxos Phase 1 for a slot."""
        prop_num = self._make_proposal_number(0)
        self._proposals[slot] = {
            'round': 0,
            'phase': 'prepare',
            'value': value,
            'original_value': value,
            'proposal_number': prop_num,
            'promises': {},
            'accepts': {},
            'highest_accepted': None,
            'highest_accepted_value': None,
        }

        # Send prepare to all nodes (including self)
        for nid in self.all_ids:
            outgoing.append({
                'type': 'prepare',
                'slot': slot,
                'proposal_number': prop_num,
                'proposer_id': self.node_id,
                'to': nid,
                'from': self.node_id,
            })

    def receive(self, msg):
        """Handle an incoming message. Returns list of outgoing messages."""
        if not self._alive:
            return []

        outgoing = []
        msg_type = msg['type']

        if msg_type == 'prepare':
            self._handle_prepare(msg, outgoing)
        elif msg_type == 'prepare_response':
            self._handle_prepare_response(msg, outgoing)
        elif msg_type == 'accept_request':
            self._handle_accept_request(msg, outgoing)
        elif msg_type == 'accept_response':
            self._handle_accept_response(msg, outgoing)
        elif msg_type == 'decided':
            self._handle_decided(msg, outgoing)

        return outgoing

    def _handle_prepare(self, msg, outgoing):
        slot = msg['slot']
        inst = self._get_instance(slot)
        result = inst.prepare(msg['proposal_number'], msg['proposer_id'])
        outgoing.append({
            'type': 'prepare_response',
            'slot': slot,
            'proposal_number': msg['proposal_number'],
            'proposer_id': msg['proposer_id'],
            'to': msg['from'],
            'from': self.node_id,
            **result,
        })

    def _handle_prepare_response(self, msg, outgoing):
        slot = msg['slot']
        if slot not in self._proposals:
            return
        prop = self._proposals[slot]
        if msg['proposal_number'] != prop['proposal_number'] or prop['phase'] != 'prepare':
            return

        if msg['promised']:
            prop['promises'][msg['from']] = True
            # Track highest accepted value
            if msg['accepted_proposal'] is not None:
                if (prop['highest_accepted'] is None or
                        msg['accepted_proposal'] > prop['highest_accepted']):
                    prop['highest_accepted'] = msg['accepted_proposal']
                    prop['highest_accepted_value'] = msg['accepted_value']

            if len(prop['promises']) >= self.majority:
                # Phase 2: send accept
                prop['phase'] = 'accept'
                # Use highest accepted value if any, else our value
                value = (prop['highest_accepted_value']
                         if prop['highest_accepted_value'] is not None
                         else prop['value'])
                prop['value'] = value
                for nid in self.all_ids:
                    outgoing.append({
                        'type': 'accept_request',
                        'slot': slot,
                        'proposal_number': prop['proposal_number'],
                        'value': value,
                        'proposer_id': self.node_id,
                        'to': nid,
                        'from': self.node_id,
                    })
        else:
            prop.setdefault('rejections', 0)
            prop['rejections'] += 1
            # Only retry when majority is impossible
            if prop['rejections'] > self.num_nodes - self.majority:
                self._bump_and_retry(slot, outgoing)

    def _handle_accept_request(self, msg, outgoing):
        slot = msg['slot']
        inst = self._get_instance(slot)
        result = inst.accept(msg['proposal_number'], msg['value'], msg['proposer_id'])
        outgoing.append({
            'type': 'accept_response',
            'slot': slot,
            'proposal_number': msg['proposal_number'],
            'value': msg['value'],
            'proposer_id': msg['proposer_id'],
            'to': msg['from'],
            'from': self.node_id,
            **result,
        })

    def _handle_accept_response(self, msg, outgoing):
        slot = msg['slot']
        if slot not in self._proposals:
            return
        prop = self._proposals[slot]
        if msg['proposal_number'] != prop['proposal_number'] or prop['phase'] != 'accept':
            return

        if msg['accepted']:
            prop['accepts'][msg['from']] = True
            if len(prop['accepts']) >= self.majority:
                # Decided!
                inst = self._get_instance(slot)
                if not inst.is_decided:
                    inst.force_decide(msg['value'])
                    # Broadcast decided to all
                    for nid in self.all_ids:
                        outgoing.append({
                            'type': 'decided',
                            'slot': slot,
                            'value': msg['value'],
                            'to': nid,
                            'from': self.node_id,
                        })
                original = prop['original_value']
                prop['phase'] = 'done'
                del self._proposals[slot]
                self._deliver_decided_slots()
                # Re-queue if our original value lost this slot
                if inst.decided_value != original:
                    self._pending.append(original)
        else:
            prop.setdefault('accept_rejections', 0)
            prop['accept_rejections'] += 1
            if prop['accept_rejections'] > self.num_nodes - self.majority:
                self._bump_and_retry(slot, outgoing)

    def _bump_and_retry(self, slot, outgoing):
        """Bump round number and retry prepare for a slot."""
        if slot not in self._proposals:
            return
        prop = self._proposals[slot]
        # If slot already decided, re-queue original value for another slot
        inst = self._get_instance(slot)
        if inst.is_decided:
            original = prop['original_value']
            del self._proposals[slot]
            if inst.decided_value != original:
                self._pending.append(original)
            return
        prop['round'] += 1
        prop_num = self._make_proposal_number(prop['round'])
        prop['proposal_number'] = prop_num
        prop['phase'] = 'prepare'
        prop['promises'] = {}
        prop['accepts'] = {}
        prop['rejections'] = 0
        prop['accept_rejections'] = 0
        prop['highest_accepted'] = None
        prop['highest_accepted_value'] = None

        for nid in self.all_ids:
            outgoing.append({
                'type': 'prepare',
                'slot': slot,
                'proposal_number': prop_num,
                'proposer_id': self.node_id,
                'to': nid,
                'from': self.node_id,
            })

    def _handle_decided(self, msg, outgoing):
        slot = msg['slot']
        inst = self._get_instance(slot)
        if not inst.is_decided:
            inst.force_decide(msg['value'])
        # If we had a proposal for this slot with a different value, re-queue it
        if slot in self._proposals:
            original = self._proposals[slot]['original_value']
            del self._proposals[slot]
            if inst.decided_value != original:
                self._pending.append(original)
        self._deliver_decided_slots()

    def crash(self):
        self._alive = False

    def recover(self, decided_slots):
        """Recover from crash, catching up on decided slots."""
        self._alive = True
        self._pending = []
        self._proposals = {}
        for slot, value in sorted(decided_slots.items()):
            inst = self._get_instance(slot)
            if not inst.is_decided:
                inst.force_decide(value)
        self._deliver_decided_slots()


class TOBCluster:
    """A cluster of TOB nodes with message routing."""

    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.nodes = {}
        for i in range(num_nodes):
            peers = [j for j in range(num_nodes) if j != i]
            self.nodes[i] = TOBNode(i, peers)

    def broadcast(self, sender_id, message):
        self.nodes[sender_id].broadcast(message)

    def run_until_delivered(self, expected_count, max_rounds=1000):
        """Run until all alive nodes have delivered expected_count messages."""
        for _ in range(max_rounds):
            # Check if done
            if self._all_delivered(expected_count):
                return True

            # Collect all outgoing messages from tick()
            all_msgs = []
            for nid, node in self.nodes.items():
                if node.is_alive:
                    msgs = node.tick()
                    all_msgs.extend(msgs)

            # Route messages
            all_msgs = self._route_messages(all_msgs)

            if not all_msgs and self._all_delivered(expected_count):
                return True

        return self._all_delivered(expected_count)

    def _route_messages(self, messages, depth=0):
        """Route messages to destination nodes, collect responses."""
        if depth > 500:
            return messages  # safety limit; remaining messages handled next tick
        new_msgs = []
        for msg in messages:
            dest = msg['to']
            if dest in self.nodes and self.nodes[dest].is_alive:
                responses = self.nodes[dest].receive(msg)
                new_msgs.extend(responses)

        # Recursively route responses until no more messages
        if new_msgs:
            return self._route_messages(new_msgs, depth + 1)
        return []

    def _all_delivered(self, expected_count):
        for node in self.nodes.values():
            if node.is_alive and len(node.delivered_messages) < expected_count:
                return False
        return True

    def get_delivery_order(self, node_id):
        return [msg for _, msg in self.nodes[node_id].delivered_messages]

    def verify_total_order(self):
        """Verify all alive nodes delivered messages in the same order."""
        alive_orders = []
        for node in self.nodes.values():
            if node.is_alive:
                alive_orders.append(self.get_delivery_order(node.node_id))
        if not alive_orders:
            return True
        # All should match the first one (up to the shortest length for partial comparison)
        min_len = min(len(o) for o in alive_orders)
        for o in alive_orders:
            if o[:min_len] != alive_orders[0][:min_len]:
                return False
        # All should have the same length
        return all(len(o) == len(alive_orders[0]) for o in alive_orders)

    def crash_node(self, node_id):
        self.nodes[node_id].crash()

    def recover_node(self, node_id):
        """Recover a crashed node, replaying all decided slots from alive nodes."""
        # Gather all decided slots from alive nodes
        decided = {}
        for node in self.nodes.values():
            if node.is_alive:
                for slot, val in node.delivered_messages:
                    decided[slot] = val
                break
        self.nodes[node_id].recover(decided)

    def get_node(self, node_id):
        return self.nodes[node_id]


class LinearizableRegister:
    """A linearizable key-value register built on total order broadcast."""

    def __init__(self, cluster, node_id):
        self.cluster = cluster
        self.node_id = node_id
        self._state = {}  # key -> value
        self._op_results = {}  # op_id -> result
        self._op_counter = 0

        # Register delivery callback on the owning node only
        cluster.get_node(node_id).on_deliver(self._on_deliver)

    def _next_op_id(self):
        self._op_counter += 1
        return f"op_{self.node_id}_{self._op_counter}"

    def _on_deliver(self, slot, message):
        """Process delivered operation."""
        if not isinstance(message, dict) or 'op_type' not in message:
            return

        op_type = message['op_type']
        op_id = message.get('op_id')
        key = message.get('key')

        if op_type == 'write':
            self._state[key] = message['value']
            if op_id:
                self._op_results[op_id] = True

        elif op_type == 'cas':
            current = self._state.get(key)
            if current == message['expected']:
                self._state[key] = message['new_value']
                if op_id:
                    self._op_results[op_id] = True
            else:
                if op_id:
                    self._op_results[op_id] = False

        elif op_type == 'read':
            if op_id:
                self._op_results[op_id] = self._state.get(key)

    def read(self, key):
        """Linearizable read."""
        op_id = self._next_op_id()
        self.cluster.broadcast(self.node_id, {
            'op_type': 'read',
            'key': key,
            'op_id': op_id,
        })
        # Count current deliveries then run
        current = len(self.cluster.get_node(self.node_id).delivered_messages)
        self.cluster.run_until_delivered(current + 1)
        return self._op_results.get(op_id)

    def write(self, key, value):
        """Write a value."""
        op_id = self._next_op_id()
        self.cluster.broadcast(self.node_id, {
            'op_type': 'write',
            'key': key,
            'value': value,
            'op_id': op_id,
        })
        current = len(self.cluster.get_node(self.node_id).delivered_messages)
        self.cluster.run_until_delivered(current + 1)

    def compare_and_set(self, key, expected, new_value):
        """Atomic compare-and-set."""
        op_id = self._next_op_id()
        self.cluster.broadcast(self.node_id, {
            'op_type': 'cas',
            'key': key,
            'expected': expected,
            'new_value': new_value,
            'op_id': op_id,
        })
        current = len(self.cluster.get_node(self.node_id).delivered_messages)
        self.cluster.run_until_delivered(current + 1)
        return self._op_results.get(op_id, False)

    def get_state(self):
        return dict(self._state)

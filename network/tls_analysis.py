#!/usr/bin/python

"""
Analyze tls datastrem.
"""

import optparse



def read_hex(bfile, num_bytes):
    return bfile.read(num_bytes).encode('hex')


def next_record_type(binary_file):
    hex_byte = read_hex(binary_file, 1)
    if hex_byte == "":
        return None
    else:
        content_type = int(hex_byte, 16)
        binary_file.seek(-1, 1) # go back one byte
        return content_type


class Record(object):

    ContentType = {
        20: "change cipher spec",
        21: "alert",
        22: "handshake",
        23: "application data",
    }

    def __init__(self, *args):

        binary_file = args[0]

        self.content_type = int(read_hex(binary_file, 1), 16)
        self.major_version = int(read_hex(binary_file, 1), 16)
        self.minor_version = int(read_hex(binary_file, 1), 16)
        self.length = int(read_hex(binary_file, 2), 16)
        self.fragment = read_hex(binary_file, self.length)
        self.fragment_location = 0

    def read_hex(self, num_bytes):
        num_chars = num_bytes*2
        byte_str = self.fragment[self.fragment_location:self.fragment_location + num_chars]
        self.fragment_location += num_chars
        return byte_str

    def __str__(self):
        out = []
        out.append("    Content Type: %s" % self.ContentType[self.content_type])
        out.append("     TLS Version: %d.%d" % (self.major_version, self.minor_version))
        out.append("   Record Length: %d" % self.length)

        return "\n".join(out)


class HandshakeRecord(Record):

    HandshakeType = {
        0: "hello request",
        1: "client hello",
        2: "server hello",
        11: "certificate",
        12: "server key exchange ",
        13: "certificate request",
        14: "server hello done",
        15: "certificate verify",
        16: "client key exchange",
        20: "finished",
    }

    def __init__(self, *args):
        super(HandshakeRecord, self).__init__(*args)

        self.handshake_type = int(self.read_hex(1), 16)
        self.handshake_length = int(self.read_hex(3), 16)
        print(self.handshake_type)


    def __str__(self):
        out = []
        out.append(super(HandshakeRecord, self).__str__())
        out.append("  Handshake Type: %s" % self.HandshakeType[self.handshake_type])
        out.append("Handshake Length: %d" % self.handshake_length)

        return "\n".join(out)


class ChangeCipherRecord(Record):
    pass


class AlertRecord(Record):
    pass


class ApplicationDataRecord(Record):
    pass


if __name__ == "__main__":

    parser = optparse.OptionParser()
    options, args = parser.parse_args()

    tcp_filename = args[0]

    tcp_file = open(tcp_filename, 'rb')

    for record_type in iter(lambda: next_record_type(tcp_file), None):
        record_type = Record.ContentType[record_type]
        if record_type == "change cipher spec":
            record = ChangeCipherRecord(tcp_file)
        elif record_type == "handshake":
            record = HandshakeRecord(tcp_file)
        elif record_type == "alert":
            record = AlertRecord(tcp_file)
        elif record_type == "application data":
            record = ApplicationDataRecord(tcp_file)
        else:
            raise Exception("Unknown content type")

        print(record)
        print("\n")

    tcp_file.close()


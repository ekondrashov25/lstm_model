import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom

from datetime import datetime

class XESLogger:
    def __init__(self):
        self.root = ET.Element("log", {
            "xes.version": "2.0",
            "xes.features": "",
            "xmlns": "http://www.xes-standard.org/"
        })
        self.add_extensions()
        self.add_classifiers()
        self.add_global_attributes()

    def add_extensions(self):
        extensions = [
            {"name": "Lifecycle", "prefix": "lifecycle", "uri": "http://www.xes-standard.org/lifecycle.xesext"},
            {"name": "Time", "prefix": "time", "uri": "http://www.xes-standard.org/time.xesext"},
            {"name": "Concept", "prefix": "concept", "uri": "http://www.xes-standard.org/concept.xesext"}
        ]
        for ext in extensions:
            ET.SubElement(self.root, "extension", ext)

    def add_classifiers(self):
        classifiers = [
            {"name": "Event Name", "keys": "concept:name"},
        ]
        for classifier in classifiers:
            ET.SubElement(self.root, "classifier", classifier)

        ET.SubElement(self.root, "string", {"key": "concept:name", "value": "lstm_training"})
        ET.SubElement(self.root, "string", {"key": "lifecycle:model", "value": "standard"})

    def add_global_attributes(self):
        global_trace = ET.SubElement(self.root, "global", {"scope": "trace"})
        ET.SubElement(global_trace, "string", {"key": "concept:name", "value": "__INVALID__"})

        global_event = ET.SubElement(self.root, "global", {"scope": "event"})
        ET.SubElement(global_event, "string", {"key": "concept:name", "value": "__INVALID__"})
        ET.SubElement(global_event, "string", {"key": "lifecycle:transition", "value": "complete"})

    def add_trace(self, trace_id):
        trace = ET.SubElement(self.root, "trace")
        ET.SubElement(trace, "string", {"key": "concept:name", "value": trace_id})
        return trace

    def add_event(self, trace, name, timestamp=None, attrs=None):
        event = ET.SubElement(trace, 'event')
        timestamp = timestamp or datetime.now().isoformat() + "+03:00"
        ET.SubElement(event, 'string', {'key': 'concept:name', 'value': name})
        ET.SubElement(event, 'string', {'key': 'lifecycle:transition', 'value': 'complete'})
        ET.SubElement(event, 'date', {'key': 'time:timestamp', 'value': timestamp})
        if attrs:
            for k, v in attrs.items():
                if isinstance(v, float):
                    ET.SubElement(event, 'float', {'key': k, 'value': str(round(v, 4))})
                elif isinstance(v, int):
                    ET.SubElement(event, 'int', {'key': k, 'value': str(v)})
                elif isinstance(v, bool):
                    ET.SubElement(event, 'int', {'key': k, 'value': str(1 if v else 0)})
                else:
                    ET.SubElement(event, 'string', {'key': k, 'value': str(v)})
    
    def add_trace_attribute(self, trace, key, value):
        ET.SubElement(trace, 'string', {'key': key, 'value': str(value)})
        

    def save_xes_file(self, filename):
        tree = ET.ElementTree(self.root)
        tree.write(filename, encoding='utf-8', xml_declaration=True)
        print(f".xes file saved to {filename}")
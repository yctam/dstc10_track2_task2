import os
import json
import re
import logging

logger = logging.getLogger(__name__)

class KnowledgeReader(object):
    def __init__(self, dataroot, knowledge_file, db_file="db.json"):
        path = os.path.join(os.path.abspath(dataroot))

        with open(os.path.join(path, knowledge_file), 'r') as f:
            self.knowledge = json.load(f)

        # load db file if any
        self.phone_map = {}
        self.address_map = {}
        try:
            logger.info("loading db.json {}".format(os.path.join(path, db_file)))
            with open(os.path.join(path, db_file), 'r') as f:
                self.db = json.load(f)
        
                for domain, entity_list in self.db.items():
                  for e in entity_list:
                    phone = e["phone"]
                    address = e["address"]
                    if phone == "":
                      continue
                    eid = e["id"]
                    # id-map key
                    key = "{}__{}".format(domain, eid)
                    self.phone_map[key] = phone
                    self.address_map[key] = address
                    #print(key, phone, address)
        except:
            logger.info("db.json has exception!")
            #print("db.json not found")
            pass

    def get_phone(self, key):
        return self.phone_map[key] if key in self.phone_map else ""

    def get_address(self, key):
        return self.address_map[key] if key in self.address_map else ""

    def get_domain_list(self):
        return list(self.knowledge.keys())

    def get_entity_list(self, domain):
        if domain not in self.get_domain_list():
            raise ValueError("invalid domain name")

        entity_ids = []
        for entity_id in self.knowledge[domain].keys():
            try:
                entity_id = int(entity_id)
                entity_ids.append(int(entity_id))
            except:
                pass

        result = []
        for entity_id in sorted(entity_ids):
            entity_name = self.knowledge[domain][str(entity_id)]['name']
            result.append({'id': entity_id, 'name': entity_name})

        return result

    def get_entity_name(self, domain, entity_id):
        if domain not in self.get_domain_list():
            raise ValueError("invalid domain name: %s" % domain)

        if str(entity_id) not in self.knowledge[domain]:
            raise ValueError("invalid entity id: %s" % str(entity_id))

        result = self.knowledge[domain][str(entity_id)]['name'] or None

        return result


    def get_doc_list(self, domain=None, entity_id=None):
        if domain is None:
            domain_list = self.get_domain_list()
        else:
            if domain not in self.get_domain_list():
                raise ValueError("invalid domain name: %s" % domain)
            domain_list = [domain]

        result = []
        for domain in domain_list:
            if entity_id is None:
                for item_id, item_obj in self.knowledge[domain].items():
                    item_name = self.get_entity_name(domain, item_id)
                    
                    if item_id != '*':
                        item_id = int(item_id)

                    for doc_id, doc_obj in item_obj['docs'].items():
                        result.append({'domain': domain, 'entity_id': item_id, 'entity_name': item_name, 'doc_id': doc_id, 'doc': {'title': doc_obj['title'], 'body': doc_obj['body']}})
            else:
                if str(entity_id) not in self.knowledge[domain]:
                    raise ValueError("invalid entity id: %s" % str(entity_id))

                entity_name = self.get_entity_name(domain, entity_id)
                
                entity_obj = self.knowledge[domain][str(entity_id)]
                for doc_id, doc_obj in entity_obj['docs'].items():
                    result.append({'domain': domain, 'entity_id': entity_id, 'entity_name': entity_name, 'doc_id': doc_id, 'doc': {'title': doc_obj['title'], 'body': doc_obj['body']}})
        return result

    def get_doc(self, domain, entity_id, doc_id):
        if domain not in self.get_domain_list():
            raise ValueError("invalid domain name: %s" % domain)

        if str(entity_id) not in self.knowledge[domain]:
            raise ValueError("invalid entity id: %s" % str(entity_id))

        entity_name = self.get_entity_name(domain, entity_id)

        if str(doc_id) not in self.knowledge[domain][str(entity_id)]['docs']:
            raise ValueError("invalid doc id: %s" % str(doc_id))

        doc_obj = self.knowledge[domain][str(entity_id)]['docs'][str(doc_id)]
        result = {'domain': domain, 'entity_id': entity_id, 'entity_name': entity_name, 'doc_id': doc_id, 'doc': {'title': doc_obj['title'], 'body': doc_obj['body']}}

        return result

    # Return not only an entity name, but also the same store in other locations
    def get_entity_name_of_various_locations(self, domain, entity_id):
        if domain not in self.get_domain_list():
            raise ValueError("invalid domain name: %s" % domain)

        if str(entity_id) not in self.knowledge[domain]:
            raise ValueError("invalid entity id: %s" % str(entity_id))

        entity_name = self.knowledge[domain][str(entity_id)]['name'] or None

        if entity_name is None:
            return []

        result = []
        # check structure of result (<name> - <location>)
        r = re.match(r"^(.+) - (.+)$", entity_name)
        if r:
            # these are same entity with different locations
            name = r[0].lower()
            loc = r[1].lower()

            # linear search over domain entities
            for entity_id in self.knowledge[domain].keys():
                try:
                    entity_id = int(entity_id)
                    name1 = self.knowledge[domain][str(entity_id)]['name']
                    r1 = re.match(r"^(.+) - (.+)$", name1)

                    # same company name
                    if r1 and r1[0].lower() == name.lower():
                        result.append(name1)
                except:
                    pass
        else:
            result = [entity_name]

        return result

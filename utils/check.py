import weaviate
from pprint import pprint

with weaviate.connect_to_local() as client:
    cols = client.collections.list_all()
    print(list(cols.keys()))

with weaviate.connect_to_local() as client:
    # 1) Show schema for TestDocs
    schema = client.collections.get("TestDocs").config.get()
    print("=== TestDocs SCHEMA ===")
    pprint(schema.to_dict())

    # 2) Count objects
    coll = client.collections.get("TestDocs")
    agg = coll.aggregate.over_all(total_count=True)
    print("\n=== OBJECT COUNT ===")
    print("Total objects:", agg.total_count)

    # 3) Peek at a few objects
    print("\n=== SAMPLE OBJECTS ===")
    res = coll.query.fetch_objects(limit=3)
    for obj in res.objects:
        pprint(obj.properties)

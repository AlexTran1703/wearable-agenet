package main

// MongoDB persistence. One document per 5-min record: metadata, the canonical waveform (int16 LE binary) and the
// AGENet analysis. Database "agenet_telemed", collection "records".

import (
	"context"
	"time"

	"go.mongodb.org/mongo-driver/v2/bson"
	"go.mongodb.org/mongo-driver/v2/mongo"
	"go.mongodb.org/mongo-driver/v2/mongo/options"
)

type Store struct {
	cli *mongo.Client
	col *mongo.Collection
}

func OpenStore(ctx context.Context, uri, db string) (*Store, error) {
	cli, err := mongo.Connect(options.Client().ApplyURI(uri).SetServerSelectionTimeout(10 * time.Second))
	if err != nil {
		return nil, err
	}
	if err := cli.Ping(ctx, nil); err != nil {
		return nil, err
	}
	col := cli.Database(db).Collection("records")
	_, _ = col.Indexes().CreateMany(ctx, []mongo.IndexModel{
		{Keys: bson.D{{Key: "device_id", Value: 1}, {Key: "nonce", Value: 1}}, Options: options.Index().SetUnique(true)},
		{Keys: bson.D{{Key: "received_at", Value: -1}}},
	})
	return &Store{cli: cli, col: col}, nil
}

func (s *Store) Ping(ctx context.Context) error { return s.cli.Ping(ctx, nil) }

func (s *Store) Insert(ctx context.Context, doc bson.M) error {
	_, err := s.col.InsertOne(ctx, doc)
	return err
}

func (s *Store) SetFields(ctx context.Context, id string, set bson.M) error {
	_, err := s.col.UpdateByID(ctx, id, bson.M{"$set": set})
	return err
}

func (s *Store) Get(ctx context.Context, id string, proj bson.M) (bson.M, error) {
	var out bson.M
	err := s.col.FindOne(ctx, bson.M{"_id": id}, options.FindOne().SetProjection(proj)).Decode(&out)
	return out, err
}

// StorageBytes reports the stored BSON size of one document (persistent storage cost per record).
func (s *Store) StorageBytes(ctx context.Context, id string) (int64, error) {
	cur, err := s.col.Aggregate(ctx, mongo.Pipeline{
		{{Key: "$match", Value: bson.M{"_id": id}}},
		{{Key: "$project", Value: bson.M{"size": bson.M{"$bsonSize": "$$ROOT"}}}},
	})
	if err != nil {
		return 0, err
	}
	defer cur.Close(ctx)
	var r []struct {
		Size int64 `bson:"size"`
	}
	if err := cur.All(ctx, &r); err != nil || len(r) == 0 {
		return 0, err
	}
	return r[0].Size, nil
}

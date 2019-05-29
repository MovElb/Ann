from marshmallow import Schema, fields, validate


class AppSchemaSection(Schema):
    debug = fields.Bool(required=True)
    environment = fields.String(
        required=True, validate=validate.OneOf(["development", "production", "testing"])
    )


class SaaSSchemaSection(Schema):
    url = fields.String(required=True)
    conn_timeout = fields.Integer(required=True)
    read_timeout = fields.Integer(required=True)


class NetSchemaSection(Schema):
    url = fields.String(required=True)
    conn_timeout = fields.Integer(required=True)
    read_timeout = fields.Integer(required=True)


class RedisSchemaSection(Schema):
    host = fields.String(required=True)
    port = fields.Integer(required=True)
    master_name = fields.String(required=True)


class ConfigSchema(Schema):
    app = fields.Nested(AppSchemaSection(), required=True)
    saas = fields.Nested(SaaSSchemaSection(), required=True)
    net = fields.Nested(NetSchemaSection(), required=True)
    redis = fields.Nested(RedisSchemaSection(), required=True)

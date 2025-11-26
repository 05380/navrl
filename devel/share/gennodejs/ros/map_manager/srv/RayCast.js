// Auto-generated. Do not edit!

// (in-package map_manager.srv)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let geometry_msgs = _finder('geometry_msgs');

//-----------------------------------------------------------


//-----------------------------------------------------------

class RayCastRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.position = null;
      this.startAngle = null;
      this.range = null;
      this.vfov_min = null;
      this.vfov_max = null;
      this.vbeams = null;
      this.hres = null;
    }
    else {
      if (initObj.hasOwnProperty('position')) {
        this.position = initObj.position
      }
      else {
        this.position = new geometry_msgs.msg.Point();
      }
      if (initObj.hasOwnProperty('startAngle')) {
        this.startAngle = initObj.startAngle
      }
      else {
        this.startAngle = 0.0;
      }
      if (initObj.hasOwnProperty('range')) {
        this.range = initObj.range
      }
      else {
        this.range = 0.0;
      }
      if (initObj.hasOwnProperty('vfov_min')) {
        this.vfov_min = initObj.vfov_min
      }
      else {
        this.vfov_min = 0.0;
      }
      if (initObj.hasOwnProperty('vfov_max')) {
        this.vfov_max = initObj.vfov_max
      }
      else {
        this.vfov_max = 0.0;
      }
      if (initObj.hasOwnProperty('vbeams')) {
        this.vbeams = initObj.vbeams
      }
      else {
        this.vbeams = 0;
      }
      if (initObj.hasOwnProperty('hres')) {
        this.hres = initObj.hres
      }
      else {
        this.hres = 0.0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type RayCastRequest
    // Serialize message field [position]
    bufferOffset = geometry_msgs.msg.Point.serialize(obj.position, buffer, bufferOffset);
    // Serialize message field [startAngle]
    bufferOffset = _serializer.float64(obj.startAngle, buffer, bufferOffset);
    // Serialize message field [range]
    bufferOffset = _serializer.float64(obj.range, buffer, bufferOffset);
    // Serialize message field [vfov_min]
    bufferOffset = _serializer.float64(obj.vfov_min, buffer, bufferOffset);
    // Serialize message field [vfov_max]
    bufferOffset = _serializer.float64(obj.vfov_max, buffer, bufferOffset);
    // Serialize message field [vbeams]
    bufferOffset = _serializer.int32(obj.vbeams, buffer, bufferOffset);
    // Serialize message field [hres]
    bufferOffset = _serializer.float64(obj.hres, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type RayCastRequest
    let len;
    let data = new RayCastRequest(null);
    // Deserialize message field [position]
    data.position = geometry_msgs.msg.Point.deserialize(buffer, bufferOffset);
    // Deserialize message field [startAngle]
    data.startAngle = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [range]
    data.range = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [vfov_min]
    data.vfov_min = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [vfov_max]
    data.vfov_max = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [vbeams]
    data.vbeams = _deserializer.int32(buffer, bufferOffset);
    // Deserialize message field [hres]
    data.hres = _deserializer.float64(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 68;
  }

  static datatype() {
    // Returns string type for a service object
    return 'map_manager/RayCastRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '14618c047ed1a962ef2052a624965300';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    geometry_msgs/Point position
    float64 startAngle
    float64 range
    float64 vfov_min
    float64 vfov_max
    int32 vbeams
    float64 hres
    
    ================================================================================
    MSG: geometry_msgs/Point
    # This contains the position of a point in free space
    float64 x
    float64 y
    float64 z
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new RayCastRequest(null);
    if (msg.position !== undefined) {
      resolved.position = geometry_msgs.msg.Point.Resolve(msg.position)
    }
    else {
      resolved.position = new geometry_msgs.msg.Point()
    }

    if (msg.startAngle !== undefined) {
      resolved.startAngle = msg.startAngle;
    }
    else {
      resolved.startAngle = 0.0
    }

    if (msg.range !== undefined) {
      resolved.range = msg.range;
    }
    else {
      resolved.range = 0.0
    }

    if (msg.vfov_min !== undefined) {
      resolved.vfov_min = msg.vfov_min;
    }
    else {
      resolved.vfov_min = 0.0
    }

    if (msg.vfov_max !== undefined) {
      resolved.vfov_max = msg.vfov_max;
    }
    else {
      resolved.vfov_max = 0.0
    }

    if (msg.vbeams !== undefined) {
      resolved.vbeams = msg.vbeams;
    }
    else {
      resolved.vbeams = 0
    }

    if (msg.hres !== undefined) {
      resolved.hres = msg.hres;
    }
    else {
      resolved.hres = 0.0
    }

    return resolved;
    }
};

class RayCastResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.points = null;
    }
    else {
      if (initObj.hasOwnProperty('points')) {
        this.points = initObj.points
      }
      else {
        this.points = [];
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type RayCastResponse
    // Serialize message field [points]
    bufferOffset = _arraySerializer.float64(obj.points, buffer, bufferOffset, null);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type RayCastResponse
    let len;
    let data = new RayCastResponse(null);
    // Deserialize message field [points]
    data.points = _arrayDeserializer.float64(buffer, bufferOffset, null)
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += 8 * object.points.length;
    return length + 4;
  }

  static datatype() {
    // Returns string type for a service object
    return 'map_manager/RayCastResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '76ce76db94343b4ddd64b9177a425ea3';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    float64[] points
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new RayCastResponse(null);
    if (msg.points !== undefined) {
      resolved.points = msg.points;
    }
    else {
      resolved.points = []
    }

    return resolved;
    }
};

module.exports = {
  Request: RayCastRequest,
  Response: RayCastResponse,
  md5sum() { return '0c76c296d09a6c11167d8e932a6ebc50'; },
  datatype() { return 'map_manager/RayCast'; }
};

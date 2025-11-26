// Auto-generated. Do not edit!

// (in-package onboard_detector.srv)


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

class GetDynamicObstaclesRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.current_position = null;
      this.range = null;
    }
    else {
      if (initObj.hasOwnProperty('current_position')) {
        this.current_position = initObj.current_position
      }
      else {
        this.current_position = new geometry_msgs.msg.Point();
      }
      if (initObj.hasOwnProperty('range')) {
        this.range = initObj.range
      }
      else {
        this.range = 0.0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type GetDynamicObstaclesRequest
    // Serialize message field [current_position]
    bufferOffset = geometry_msgs.msg.Point.serialize(obj.current_position, buffer, bufferOffset);
    // Serialize message field [range]
    bufferOffset = _serializer.float64(obj.range, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type GetDynamicObstaclesRequest
    let len;
    let data = new GetDynamicObstaclesRequest(null);
    // Deserialize message field [current_position]
    data.current_position = geometry_msgs.msg.Point.deserialize(buffer, bufferOffset);
    // Deserialize message field [range]
    data.range = _deserializer.float64(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 32;
  }

  static datatype() {
    // Returns string type for a service object
    return 'onboard_detector/GetDynamicObstaclesRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '2d85907f898a3f1571d2bcd6d1effd72';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    geometry_msgs/Point current_position
    float64 range
    
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
    const resolved = new GetDynamicObstaclesRequest(null);
    if (msg.current_position !== undefined) {
      resolved.current_position = geometry_msgs.msg.Point.Resolve(msg.current_position)
    }
    else {
      resolved.current_position = new geometry_msgs.msg.Point()
    }

    if (msg.range !== undefined) {
      resolved.range = msg.range;
    }
    else {
      resolved.range = 0.0
    }

    return resolved;
    }
};

class GetDynamicObstaclesResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.position = null;
      this.velocity = null;
      this.size = null;
    }
    else {
      if (initObj.hasOwnProperty('position')) {
        this.position = initObj.position
      }
      else {
        this.position = [];
      }
      if (initObj.hasOwnProperty('velocity')) {
        this.velocity = initObj.velocity
      }
      else {
        this.velocity = [];
      }
      if (initObj.hasOwnProperty('size')) {
        this.size = initObj.size
      }
      else {
        this.size = [];
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type GetDynamicObstaclesResponse
    // Serialize message field [position]
    // Serialize the length for message field [position]
    bufferOffset = _serializer.uint32(obj.position.length, buffer, bufferOffset);
    obj.position.forEach((val) => {
      bufferOffset = geometry_msgs.msg.Vector3.serialize(val, buffer, bufferOffset);
    });
    // Serialize message field [velocity]
    // Serialize the length for message field [velocity]
    bufferOffset = _serializer.uint32(obj.velocity.length, buffer, bufferOffset);
    obj.velocity.forEach((val) => {
      bufferOffset = geometry_msgs.msg.Vector3.serialize(val, buffer, bufferOffset);
    });
    // Serialize message field [size]
    // Serialize the length for message field [size]
    bufferOffset = _serializer.uint32(obj.size.length, buffer, bufferOffset);
    obj.size.forEach((val) => {
      bufferOffset = geometry_msgs.msg.Vector3.serialize(val, buffer, bufferOffset);
    });
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type GetDynamicObstaclesResponse
    let len;
    let data = new GetDynamicObstaclesResponse(null);
    // Deserialize message field [position]
    // Deserialize array length for message field [position]
    len = _deserializer.uint32(buffer, bufferOffset);
    data.position = new Array(len);
    for (let i = 0; i < len; ++i) {
      data.position[i] = geometry_msgs.msg.Vector3.deserialize(buffer, bufferOffset)
    }
    // Deserialize message field [velocity]
    // Deserialize array length for message field [velocity]
    len = _deserializer.uint32(buffer, bufferOffset);
    data.velocity = new Array(len);
    for (let i = 0; i < len; ++i) {
      data.velocity[i] = geometry_msgs.msg.Vector3.deserialize(buffer, bufferOffset)
    }
    // Deserialize message field [size]
    // Deserialize array length for message field [size]
    len = _deserializer.uint32(buffer, bufferOffset);
    data.size = new Array(len);
    for (let i = 0; i < len; ++i) {
      data.size[i] = geometry_msgs.msg.Vector3.deserialize(buffer, bufferOffset)
    }
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += 24 * object.position.length;
    length += 24 * object.velocity.length;
    length += 24 * object.size.length;
    return length + 12;
  }

  static datatype() {
    // Returns string type for a service object
    return 'onboard_detector/GetDynamicObstaclesResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '5751cdcce82bba0007c77d0aa5f1b7cc';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    geometry_msgs/Vector3[] position
    geometry_msgs/Vector3[] velocity
    geometry_msgs/Vector3[] size
    
    ================================================================================
    MSG: geometry_msgs/Vector3
    # This represents a vector in free space. 
    # It is only meant to represent a direction. Therefore, it does not
    # make sense to apply a translation to it (e.g., when applying a 
    # generic rigid transformation to a Vector3, tf2 will only apply the
    # rotation). If you want your data to be translatable too, use the
    # geometry_msgs/Point message instead.
    
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
    const resolved = new GetDynamicObstaclesResponse(null);
    if (msg.position !== undefined) {
      resolved.position = new Array(msg.position.length);
      for (let i = 0; i < resolved.position.length; ++i) {
        resolved.position[i] = geometry_msgs.msg.Vector3.Resolve(msg.position[i]);
      }
    }
    else {
      resolved.position = []
    }

    if (msg.velocity !== undefined) {
      resolved.velocity = new Array(msg.velocity.length);
      for (let i = 0; i < resolved.velocity.length; ++i) {
        resolved.velocity[i] = geometry_msgs.msg.Vector3.Resolve(msg.velocity[i]);
      }
    }
    else {
      resolved.velocity = []
    }

    if (msg.size !== undefined) {
      resolved.size = new Array(msg.size.length);
      for (let i = 0; i < resolved.size.length; ++i) {
        resolved.size[i] = geometry_msgs.msg.Vector3.Resolve(msg.size[i]);
      }
    }
    else {
      resolved.size = []
    }

    return resolved;
    }
};

module.exports = {
  Request: GetDynamicObstaclesRequest,
  Response: GetDynamicObstaclesResponse,
  md5sum() { return 'a508c5cda827e9832476a1de6a1e2f79'; },
  datatype() { return 'onboard_detector/GetDynamicObstacles'; }
};

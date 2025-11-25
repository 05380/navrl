// Auto-generated. Do not edit!

// (in-package map_manager.srv)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------


//-----------------------------------------------------------

class CheckPosCollisionRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.x = null;
      this.y = null;
      this.z = null;
      this.inflated = null;
    }
    else {
      if (initObj.hasOwnProperty('x')) {
        this.x = initObj.x
      }
      else {
        this.x = 0.0;
      }
      if (initObj.hasOwnProperty('y')) {
        this.y = initObj.y
      }
      else {
        this.y = 0.0;
      }
      if (initObj.hasOwnProperty('z')) {
        this.z = initObj.z
      }
      else {
        this.z = 0.0;
      }
      if (initObj.hasOwnProperty('inflated')) {
        this.inflated = initObj.inflated
      }
      else {
        this.inflated = false;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type CheckPosCollisionRequest
    // Serialize message field [x]
    bufferOffset = _serializer.float64(obj.x, buffer, bufferOffset);
    // Serialize message field [y]
    bufferOffset = _serializer.float64(obj.y, buffer, bufferOffset);
    // Serialize message field [z]
    bufferOffset = _serializer.float64(obj.z, buffer, bufferOffset);
    // Serialize message field [inflated]
    bufferOffset = _serializer.bool(obj.inflated, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type CheckPosCollisionRequest
    let len;
    let data = new CheckPosCollisionRequest(null);
    // Deserialize message field [x]
    data.x = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [y]
    data.y = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [z]
    data.z = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [inflated]
    data.inflated = _deserializer.bool(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 25;
  }

  static datatype() {
    // Returns string type for a service object
    return 'map_manager/CheckPosCollisionRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '4fcaedfaf867530b4bba38a583aa9a92';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    float64 x
    float64 y
    float64 z
    bool inflated
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new CheckPosCollisionRequest(null);
    if (msg.x !== undefined) {
      resolved.x = msg.x;
    }
    else {
      resolved.x = 0.0
    }

    if (msg.y !== undefined) {
      resolved.y = msg.y;
    }
    else {
      resolved.y = 0.0
    }

    if (msg.z !== undefined) {
      resolved.z = msg.z;
    }
    else {
      resolved.z = 0.0
    }

    if (msg.inflated !== undefined) {
      resolved.inflated = msg.inflated;
    }
    else {
      resolved.inflated = false
    }

    return resolved;
    }
};

class CheckPosCollisionResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.occupied = null;
    }
    else {
      if (initObj.hasOwnProperty('occupied')) {
        this.occupied = initObj.occupied
      }
      else {
        this.occupied = false;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type CheckPosCollisionResponse
    // Serialize message field [occupied]
    bufferOffset = _serializer.bool(obj.occupied, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type CheckPosCollisionResponse
    let len;
    let data = new CheckPosCollisionResponse(null);
    // Deserialize message field [occupied]
    data.occupied = _deserializer.bool(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 1;
  }

  static datatype() {
    // Returns string type for a service object
    return 'map_manager/CheckPosCollisionResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '44d7026c7dcc8daf286cc35a49a8f442';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    bool occupied
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new CheckPosCollisionResponse(null);
    if (msg.occupied !== undefined) {
      resolved.occupied = msg.occupied;
    }
    else {
      resolved.occupied = false
    }

    return resolved;
    }
};

module.exports = {
  Request: CheckPosCollisionRequest,
  Response: CheckPosCollisionResponse,
  md5sum() { return '5b2617caaa5faa9b1c910b8c4d0cf2ea'; },
  datatype() { return 'map_manager/CheckPosCollision'; }
};

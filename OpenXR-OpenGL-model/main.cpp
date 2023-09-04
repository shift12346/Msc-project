// Copyright (c) 2017-2020 The Khronos Group Inc.
// Copyright (c) 2020 ReliaSolve LLC.
//
// SPDX-License-Identifier: Apache-2.0
#define STB_IMAGE_IMPLEMENTATION
//#include "F:/VR/OpenXR/OpenXR-OpenGL-model/build/modelLoader.hpp"
//#include <glad/glad.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <array>
#include <map>
#include <fstream>
#include <vector>
#include <string>
#include <stb_image.h>
#include "pch.h"
#include "common.h"
#include "gfxwrapper_opengl.h"
#include "xr_linear.h"
#include <opencv2/opencv.hpp>
#include <ctime>
#include "F:/VR/OpenXR/OpenXR-OpenGL-Example/build/PixMix.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>


//####################################mesh
#define MAX_BONE_INFLUENCE 4

struct Vertex {
    // position
    glm::vec3 Position;
    // normal
    glm::vec3 Normal;
    // texCoords
    glm::vec2 TexCoords;
    // tangent
    glm::vec3 Tangent;
    // bitangent
    glm::vec3 Bitangent;
    //bone indexes which will influence this vertex
    int m_BoneIDs[MAX_BONE_INFLUENCE];
    //weights from each bone
    float m_Weights[MAX_BONE_INFLUENCE];
};

struct Texture {
    unsigned int id;
    std::string type;
    std::string path;
};

class Mesh {
public:
    // mesh Data
    std::vector<Vertex>       vertices;
    std::vector<unsigned int> indices;
    std::vector<Texture>      textures;
    unsigned int VAO;

    // constructor
    Mesh(std::vector<Vertex> vertices, std::vector<unsigned int> indices, std::vector<Texture> textures)
    {
        this->vertices = vertices;
        this->indices = indices;
        this->textures = textures;

        // now that we have all the required data, set the vertex buffers and its attribute pointers.
        setupMesh();
    }

    // render the mesh
    void Draw(GLuint& shaderID)
    {
        // bind appropriate textures
        unsigned int diffuseNr = 1;
        unsigned int specularNr = 1;
        unsigned int normalNr = 1;
        unsigned int heightNr = 1;
        for (unsigned int i = 0; i < textures.size(); i++)
        {
            glActiveTexture(GL_TEXTURE0 + i); // active proper texture unit before binding
            // retrieve texture number (the N in diffuse_textureN)
            std::string number;
            std::string name = textures[i].type;
            if (name == "texture_diffuse")
                number = std::to_string(diffuseNr++);
            else if (name == "texture_specular")
                number = std::to_string(specularNr++); // transfer unsigned int to string
            else if (name == "texture_normal")
                number = std::to_string(normalNr++); // transfer unsigned int to string
            else if (name == "texture_height")
                number = std::to_string(heightNr++); // transfer unsigned int to string

            // now set the sampler to the correct texture unit
            glUniform1i(glGetUniformLocation(shaderID, (name + number).c_str()), i);
            // and finally bind the texture
            glBindTexture(GL_TEXTURE_2D, textures[i].id);
        }

        // draw mesh
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, static_cast<unsigned int>(indices.size()), GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);

        // always good practice to set everything back to defaults once configured.
        glActiveTexture(GL_TEXTURE0);
    }

private:
    // render data 
    unsigned int VBO, EBO;

    // initializes all the buffer objects/arrays
    void setupMesh()
    {
        // create buffers/arrays
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &EBO);

        glBindVertexArray(VAO);
        // load data into vertex buffers
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), &vertices[0], GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);

        // set the vertex attribute pointers
        // vertex Positions
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
        // vertex normals
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Normal));
        // vertex texture coords
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, TexCoords));
        // vertex tangent
        glEnableVertexAttribArray(3);
        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Tangent));
        // vertex bitangent
        glEnableVertexAttribArray(4);
        glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Bitangent));
        // ids
        glEnableVertexAttribArray(5);
        glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, m_BoneIDs));

        // weights
        glEnableVertexAttribArray(6);
        glVertexAttribPointer(6, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, m_Weights));
        glBindVertexArray(0);
    }
};



//####################################mesh

unsigned int TextureFromFile(const char* path, const std::string& directory, bool gamma = false);

class Model
{
public:
    // model data 
    std::vector<Texture> textures_loaded;	// stores all the textures loaded so far, optimization to make sure textures aren't loaded more than once.
    std::vector<Mesh>    meshes;
    std::string directory;
    bool gammaCorrection;

    // constructor, expects a filepath to a 3D model.
    Model(std::string const& path, bool gamma = false) : gammaCorrection(gamma)
    {
        loadModel(path);
    }

    // draws the model, and thus all its meshes
    void Draw(GLuint& shaderID)
    {
        for (unsigned int i = 0; i < meshes.size(); i++)
            meshes[i].Draw(shaderID);
    }

private:
    // loads a model with supported ASSIMP extensions from file and stores the resulting meshes in the meshes vector.
    void loadModel(std::string const& path)
    {
        // read file via ASSIMP
        Assimp::Importer importer;
        const aiScene* scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_FlipUVs | aiProcess_CalcTangentSpace);
        // check for errors
        if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) // if is Not Zero
        {
            std::cout << "ERROR::ASSIMP:: " << importer.GetErrorString() << std::endl;
            return;
        }
        // retrieve the directory path of the filepath
        directory = path.substr(0, path.find_last_of('/'));

        // process ASSIMP's root node recursively
        processNode(scene->mRootNode, scene);
    }

    // processes a node in a recursive fashion. Processes each individual mesh located at the node and repeats this process on its children nodes (if any).
    void processNode(aiNode* node, const aiScene* scene)
    {
        // process each mesh located at the current node
        for (unsigned int i = 0; i < node->mNumMeshes; i++)
        {
            // the node object only contains indices to index the actual objects in the scene. 
            // the scene contains all the data, node is just to keep stuff organized (like relations between nodes).
            aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
            meshes.push_back(processMesh(mesh, scene));
        }
        // after we've processed all of the meshes (if any) we then recursively process each of the children nodes
        for (unsigned int i = 0; i < node->mNumChildren; i++)
        {
            processNode(node->mChildren[i], scene);
        }

    }

    Mesh processMesh(aiMesh* mesh, const aiScene* scene)
    {

        // Data containers
        std::vector<Vertex> vertices;
        std::vector<unsigned int> indices;
        std::vector<Texture> textures;

        // Helper lambda function to convert aiVector3D to glm::vec3
        auto convertToGlmVec3 = [](const aiVector3D& vec) {
            return glm::vec3(vec.x, vec.y, vec.z);
        };

        // Helper lambda function to convert aiVector3D to glm::vec2 (only first two coordinates)
        auto convertToGlmVec2 = [](const aiVector3D& vec) {
            return glm::vec2(vec.x, vec.y);
        };

        // Process vertices
        for (unsigned int i = 0; i < mesh->mNumVertices; i++) {
            Vertex vertex;
            vertex.Position = convertToGlmVec3(mesh->mVertices[i]);

            if (mesh->HasNormals()) {
                vertex.Normal = convertToGlmVec3(mesh->mNormals[i]);
            }

            if (mesh->mTextureCoords[0]) {
                vertex.TexCoords = convertToGlmVec2(mesh->mTextureCoords[0][i]);
                vertex.Tangent = convertToGlmVec3(mesh->mTangents[i]);
                vertex.Bitangent = convertToGlmVec3(mesh->mBitangents[i]);
            }
            else {
                vertex.TexCoords = glm::vec2(0.0f, 0.0f);
            }

            vertices.push_back(vertex);
        }

        // Process indices
        for (unsigned int i = 0; i < mesh->mNumFaces; i++) {
            aiFace face = mesh->mFaces[i];
            for (unsigned int j = 0; j < face.mNumIndices; j++) {
                indices.push_back(face.mIndices[j]);
            }
        }

        // Helper function to load and append textures
        auto loadAndAppendTextures = [&](aiMaterial* mat, aiTextureType type, const std::string& typeName) {
            std::vector<Texture> texMaps = loadMaterialTextures(mat, type, typeName);
            textures.insert(textures.end(), texMaps.begin(), texMaps.end());
        };

        // Process materials
        aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];
        loadAndAppendTextures(material, aiTextureType_DIFFUSE, "texture_diffuse");
        loadAndAppendTextures(material, aiTextureType_SPECULAR, "texture_specular");
        loadAndAppendTextures(material, aiTextureType_HEIGHT, "texture_normal");
        loadAndAppendTextures(material, aiTextureType_AMBIENT, "texture_height");

        // Return mesh object
        return Mesh(vertices, indices, textures);

    }

    // checks all material textures of a given type and loads the textures if they're not loaded yet.
    // the required info is returned as a Texture struct.
    std::vector<Texture> loadMaterialTextures(aiMaterial* mat, aiTextureType type, std::string typeName)
    {
        std::vector<Texture> textures;
        for (unsigned int i = 0; i < mat->GetTextureCount(type); i++)
        {
            aiString str;
            mat->GetTexture(type, i, &str);
            // check if texture was loaded before and if so, continue to next iteration: skip loading a new texture
            bool skip = false;
            for (unsigned int j = 0; j < textures_loaded.size(); j++)
            {
                if (std::strcmp(textures_loaded[j].path.data(), str.C_Str()) == 0)
                {
                    textures.push_back(textures_loaded[j]);
                    skip = true; // a texture with the same filepath has already been loaded, continue to next one. (optimization)
                    break;
                }
            }
            if (!skip)
            {   // if texture hasn't been loaded already, load it
                Texture texture;
                texture.id = TextureFromFile(str.C_Str(), this->directory);
                texture.type = typeName;
                texture.path = str.C_Str();
                textures.push_back(texture);
                textures_loaded.push_back(texture);  // store it as texture loaded for entire model, to ensure we won't unnecessary load duplicate textures.
            }
        }
        return textures;
    }
};


unsigned int TextureFromFile(const char* path, const std::string& directory, bool gamma)
{
    std::string filename = std::string(path);
    filename = directory + '/' + filename;

    unsigned int textureID;
    glGenTextures(1, &textureID);

    int width, height, nrComponents;
    stbi_set_flip_vertically_on_load(true);
    unsigned char* data = stbi_load(filename.c_str(), &width, &height, &nrComponents, 0);
    if (data)
    {
        GLenum format;
        if (nrComponents == 1)
            format = GL_RED;
        else if (nrComponents == 3)
            format = GL_RGB;
        else if (nrComponents == 4)
            format = GL_RGBA;

        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        stbi_image_free(data);
    }
    else
    {
        std::cout << "Texture failed to load at path: " << path << std::endl;
        stbi_image_free(data);
    }

    return textureID;
}
//###################################loader

// Include the file that describes the cubes we draw in this example.
#include "geometry.h"

unsigned g_verbosity = 1;

bool mode = 1;
//Model ourModel("F:/VR/OpenXR/OpenXR-OpenGL-model/skybox/objects/backpack/backpack.obj");
static void Usage(std::string name)
{

    std::cout << "Usage: " << name << " [--verbosity V]" << std::endl;
    std::cout << "       --verbosity: Set V to 0 for silence, higher for more info (default " << g_verbosity << ")" << std::endl;
}

//inpainting
static const float a(0.073235f);
static const float b(0.176765f);
static const cv::Mat K = (cv::Mat_<float>(3, 3) << a, b, a, b, 0.0f, b, a, b, a);

void inpaint(const cv::Mat& src, const cv::Mat& mask, const cv::Mat kernel, cv::Mat& dst, int maxNumOfIter = 100)
{
    assert(src.type() == mask.type() && mask.type() == CV_8UC3);
    assert(src.size() == mask.size());
    assert(kernel.type() == CV_32F);

    // fill in the missing region with the input's average color
    auto avgColor = cv::sum(src) / (src.cols * src.rows);
    cv::Mat avgColorMat(1, 1, CV_8UC3);
    avgColorMat.at<cv::Vec3b>(0, 0) = cv::Vec3b(avgColor[0], avgColor[1], avgColor[2]);
    cv::resize(avgColorMat, avgColorMat, src.size(), 0.0, 0.0, cv::INTER_NEAREST);
    cv::Mat result = (mask / 255).mul(src) + (1 - mask / 255).mul(avgColorMat);

    // convolution
    int bSize = K.cols / 2;
    cv::Mat kernel3ch, inWithBorder;
    result.convertTo(result, CV_32FC3);
    cv::cvtColor(kernel, kernel3ch, cv::COLOR_GRAY2BGR);

    cv::copyMakeBorder(result, inWithBorder, bSize, bSize, bSize, bSize, cv::BORDER_REPLICATE);
    cv::Mat resInWithBorder = cv::Mat(inWithBorder, cv::Rect(bSize, bSize, result.cols, result.rows));

    const int ch = result.channels();
    for (int itr = 0; itr < maxNumOfIter; ++itr)
    {
        cv::copyMakeBorder(result, inWithBorder, bSize, bSize, bSize, bSize, cv::BORDER_REPLICATE);

        for (int r = 0; r < result.rows; ++r)
        {
            const uchar* pMask = mask.ptr(r);
            float* pRes = result.ptr<float>(r);
            for (int c = 0; c < result.cols; ++c)
            {
                if (pMask[ch * c] == 0)
                {
                    cv::Rect rectRoi(c, r, K.cols, K.rows);
                    cv::Mat roi(inWithBorder, rectRoi);

                    auto sum = cv::sum(kernel3ch.mul(roi));
                    pRes[ch * c + 0] = sum[0];
                    pRes[ch * c + 1] = sum[1];
                    pRes[ch * c + 2] = sum[2];
                }
            }
        }

        // for debugging
        cv::imshow("Inpainting...", result / 255.0f);
        cv::waitKey(1);
    }

    result.convertTo(dst, CV_8UC3);
}


cv::Mat create_right_eye_image(const cv::Mat& image, const cv::Mat& depth_map, int max_shift)
{
    int height = image.rows;
    int width = image.cols;

    // Convert depth map to float and normalize it to range [0, 1]
    cv::Mat depth_map_float;
    depth_map.convertTo(depth_map_float, CV_32F, 1.0 / 255.0);

    // Create a disparity map based on depth
    cv::Mat disparity_map = (1.0 - depth_map_float) * max_shift;

    // Create an empty image with white pixels
    cv::Mat new_image(image.size(), image.type(), cv::Scalar(255, 255, 255));  // <-- MODIFIED LINE

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            int shift = static_cast<int>(disparity_map.at<float>(y, x));

            // Prevent indexing beyond image bounds
            if (x - shift < 0)
                continue;

            // Shift pixel
            new_image.at<cv::Vec3b>(y, x - shift) = image.at<cv::Vec3b>(y, x);
        }
    }

    return new_image;
}

cv::Mat create_right_eye_image2(const cv::Mat& image, const cv::Mat& depth_map, int max_shift)
{
    int height = image.rows;
    int width = image.cols;

    // Convert depth map to float and normalize it to range [0, 1]
    cv::Mat depth_map_float;
    depth_map.convertTo(depth_map_float, CV_32F, 1.0 / 255.0);

    // Create a disparity map based on depth
    cv::Mat disparity_map = (1.0 - depth_map_float) * max_shift;

    // Create an empty image with white pixels
    cv::Mat new_image(image.size(), image.type(), cv::Scalar(255, 255, 255));

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            int shift = static_cast<int>(disparity_map.at<float>(y, x));

            // Prevent indexing beyond image bounds
            if (x - shift < 0)
                continue;

            // Shift pixel
            new_image.at<cv::Vec3b>(y, x - shift) = image.at<cv::Vec3b>(y, x);
        }
    }

    // Gap-filling: For each white pixel, find the nearest left or right non-white pixel and use its value.
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            if (new_image.at<cv::Vec3b>(y, x) == cv::Vec3b(255, 255, 255))  // If pixel is white
            {
                // Find nearest non-white pixel to the left
                int left = x - 1;
                while (left >= 0 && new_image.at<cv::Vec3b>(y, left) == cv::Vec3b(255, 255, 255))
                {
                    --left;
                }

                // Find nearest non-white pixel to the right
                int right = x + 1;
                while (right < width && new_image.at<cv::Vec3b>(y, right) == cv::Vec3b(255, 255, 255))
                {
                    ++right;
                }

                // Choose the nearest one and use its value to fill the gap
                if (left >= 0 && right < width)
                {
                    if (x - left < right - x)
                    {
                        new_image.at<cv::Vec3b>(y, x) = new_image.at<cv::Vec3b>(y, left);
                    }
                    else
                    {
                        new_image.at<cv::Vec3b>(y, x) = new_image.at<cv::Vec3b>(y, right);
                    }
                }
                else if (left >= 0)
                {
                    new_image.at<cv::Vec3b>(y, x) = new_image.at<cv::Vec3b>(y, left);
                }
                else if (right < width)
                {
                    new_image.at<cv::Vec3b>(y, x) = new_image.at<cv::Vec3b>(y, right);
                }
            }
        }
    }

    return new_image;
}


cv::Mat warp_image_based_on_depth(const cv::Mat& image, const cv::Mat& depth_map, int max_shift)
{
    int height = image.rows;
    int width = image.cols;

    // Convert depth map to float and normalize it to range [0, 1]
    cv::Mat depth_map_float;
    depth_map.convertTo(depth_map_float, CV_32F, 1.0 / 255.0);

    // Create a disparity map based on depth
    cv::Mat disparity_map = (1.0 - depth_map_float) * max_shift;

    // Create an empty image with white pixels
    cv::Mat warped_image(image.size(), image.type(), cv::Scalar(255, 255, 255));

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            int shift = static_cast<int>(disparity_map.at<float>(y, x));

            // Prevent indexing beyond image bounds
            if (x - shift < 0)
                continue;

            // Shift pixel
            warped_image.at<cv::Vec3b>(y, x - shift) = image.at<cv::Vec3b>(y, x);
        }
    }

    return warped_image;
}

// Function for inpainting gaps in the warped image
cv::Mat inpaint_warped_image_withNearPixel(const cv::Mat& warped_image)
{
    cv::Mat inpainted_image = warped_image.clone();
    int height = inpainted_image.rows;
    int width = inpainted_image.cols;

    // Gap detection and filling
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            cv::Vec3b pixel = inpainted_image.at<cv::Vec3b>(y, x);

            // Check if pixel is white (indicating a gap)
            if (pixel == cv::Vec3b(255, 255, 255))
            {
                int left_x = x;
                while (left_x > 0 && inpainted_image.at<cv::Vec3b>(y, left_x) == cv::Vec3b(255, 255, 255))
                    left_x--;

                int right_x = x;
                while (right_x < width - 1 && inpainted_image.at<cv::Vec3b>(y, right_x) == cv::Vec3b(255, 255, 255))
                    right_x++;

                // Choose the nearest valid pixel to fill the gap
                if (abs(left_x - x) < abs(right_x - x))
                    inpainted_image.at<cv::Vec3b>(y, x) = inpainted_image.at<cv::Vec3b>(y, left_x);
                else
                    inpainted_image.at<cv::Vec3b>(y, x) = inpainted_image.at<cv::Vec3b>(y, right_x);
            }
        }
    }

    return inpainted_image;
}




//============================================================================================
// Helper functions.



namespace Math {
namespace Pose {
XrPosef Identity() {
    XrPosef t{};
    t.orientation.w = 1;
    return t;
}

XrPosef Translation(const XrVector3f& translation) {
    XrPosef t = Identity();
    t.position = translation;
    return t;
}

XrPosef RotateCCWAboutYAxis(float radians, XrVector3f translation) {
    XrPosef t = Identity();
    t.orientation.x = 0.f;
    t.orientation.y = std::sin(radians * 0.5f);
    t.orientation.z = 0.f;
    t.orientation.w = std::cos(radians * 0.5f);
    t.position = translation;
    return t;
}
}  // namespace Pose
}  // namespace Math

//============================================================================================
// Code to handle knowing which spaces things are rendered in.
/// @todo If you only want to render in world space, all of the space-choosing machinery
/// can be removed.  Note that the hands are spaces in addition to the application-defined ones.

// Maps from the space back to its name so we can know what to render in each
std::map<XrSpace, std::string> g_spaceNames;

// Description of one of the spaces we want to render in, along with a scale factor to
// be applied in that space.  In the original example, this is used to position, orient,
// and scale cubes to various spaces including hand space.
struct Space {
    XrPosef Pose;           ///< Pose of the space relative to g_appSpace
    XrVector3f Scale;       ///< Scale hint for the space
    std::string Name;       ///< An identifier so we can know what to render in each space
};

//===================================================================================
//OpenXR variables
#if !defined(XR_USE_PLATFORM_WIN32)
#define strcpy_s(dest, source) strncpy((dest), (source), sizeof(dest))
#endif

namespace Side {
    const int LEFT = 0;
    const int RIGHT = 1;
    const int COUNT = 2;
}  // namespace Side

struct InputState {
    XrActionSet actionSet{ XR_NULL_HANDLE };
    XrAction grabAction{ XR_NULL_HANDLE };
    XrAction poseAction{ XR_NULL_HANDLE };
    XrAction vibrateAction{ XR_NULL_HANDLE };
    XrAction quitAction{ XR_NULL_HANDLE };
    std::array<XrPath, Side::COUNT> handSubactionPath;
    std::array<XrSpace, Side::COUNT> handSpace;
    std::array<float, Side::COUNT> handScale = { {1.0f, 1.0f} };
    std::array<XrBool32, Side::COUNT> handActive;
};

struct Swapchain {
    XrSwapchain handle;
    int32_t width;
    int32_t height;
};

XrInstance g_instance{ XR_NULL_HANDLE };
XrSession g_session{ XR_NULL_HANDLE };
XrSpace g_appSpace{ XR_NULL_HANDLE };
XrFormFactor g_formFactor{ XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY };
XrViewConfigurationType g_viewConfigType{ XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO };
XrEnvironmentBlendMode g_environmentBlendMode{ XR_ENVIRONMENT_BLEND_MODE_OPAQUE };
XrSystemId g_systemId{ XR_NULL_SYSTEM_ID };

std::vector<XrViewConfigurationView> g_configViews;
std::vector<Swapchain> g_swapchains;
std::map<XrSwapchain, std::vector<XrSwapchainImageBaseHeader*>> g_swapchainImages;
std::vector<XrView> g_views;
int64_t g_colorSwapchainFormat{ -1 };

std::vector<XrSpace> g_visualizedSpaces;

// Application's current lifecycle state according to the runtime
XrSessionState g_sessionState{ XR_SESSION_STATE_UNKNOWN };
bool g_sessionRunning{ false };

XrEventDataBuffer g_eventDataBuffer;
InputState g_input;

//create skybox
float skyboxVertices[] = {
    // positions          
    -1.0f,  1.0f, -1.0f,
    -1.0f, -1.0f, -1.0f,
     1.0f, -1.0f, -1.0f,
     1.0f, -1.0f, -1.0f,
     1.0f,  1.0f, -1.0f,
    -1.0f,  1.0f, -1.0f,

    -1.0f, -1.0f,  1.0f,
    -1.0f, -1.0f, -1.0f,
    -1.0f,  1.0f, -1.0f,
    -1.0f,  1.0f, -1.0f,
    -1.0f,  1.0f,  1.0f,
    -1.0f, -1.0f,  1.0f,

     1.0f, -1.0f, -1.0f,
     1.0f, -1.0f,  1.0f,
     1.0f,  1.0f,  1.0f,
     1.0f,  1.0f,  1.0f,
     1.0f,  1.0f, -1.0f,
     1.0f, -1.0f, -1.0f,

    -1.0f, -1.0f,  1.0f,
    -1.0f,  1.0f,  1.0f,
     1.0f,  1.0f,  1.0f,
     1.0f,  1.0f,  1.0f,
     1.0f, -1.0f,  1.0f,
    -1.0f, -1.0f,  1.0f,

    -1.0f,  1.0f, -1.0f,
     1.0f,  1.0f, -1.0f,
     1.0f,  1.0f,  1.0f,
     1.0f,  1.0f,  1.0f,
    -1.0f,  1.0f,  1.0f,
    -1.0f,  1.0f, -1.0f,

    -1.0f, -1.0f, -1.0f,
    -1.0f, -1.0f,  1.0f,
     1.0f, -1.0f, -1.0f,
     1.0f, -1.0f, -1.0f,
    -1.0f, -1.0f,  1.0f,
     1.0f, -1.0f,  1.0f
};



// draw a full screen square that could 
float vertices[] = {
    // positions          // colors           // texture coords
     1.0f,  1.0f, 0.0f,   1.0f, 0.0f, 0.0f,   1.0f, 1.0f, // top right
     1.0f, -1.0f, 0.0f,   0.0f, 1.0f, 0.0f,   1.0f, 0.0f, // bottom right
    -1.0f, -1.0f, 0.0f,   0.0f, 0.0f, 1.0f,   0.0f, 0.0f, // bottom left
    -1.0f,  1.0f, 0.0f,   1.0f, 1.0f, 0.0f,   0.0f, 1.0f  // top left 
};

unsigned int indices[] = {
    0, 1, 3, // first triangle
    1, 2, 3  // second triangle
};

// data for the first scene
float Scene1vertices[] = {
    -0.5f, -0.5f, -0.5f,  0.0f, 0.0f,
     0.5f, -0.5f, -0.5f,  1.0f, 0.0f,
     0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
     0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
    -0.5f,  0.5f, -0.5f,  0.0f, 1.0f,
    -0.5f, -0.5f, -0.5f,  0.0f, 0.0f,

    -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
     0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
     0.5f,  0.5f,  0.5f,  1.0f, 1.0f,
     0.5f,  0.5f,  0.5f,  1.0f, 1.0f,
    -0.5f,  0.5f,  0.5f,  0.0f, 1.0f,
    -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,

    -0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
    -0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
    -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
    -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
    -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
    -0.5f,  0.5f,  0.5f,  1.0f, 0.0f,

     0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
     0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
     0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
     0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
     0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
     0.5f,  0.5f,  0.5f,  1.0f, 0.0f,

    -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
     0.5f, -0.5f, -0.5f,  1.0f, 1.0f,
     0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
     0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
    -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
    -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,

    -0.5f,  0.5f, -0.5f,  0.0f, 1.0f,
     0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
     0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
     0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
    -0.5f,  0.5f,  0.5f,  0.0f, 0.0f,
    -0.5f,  0.5f, -0.5f,  0.0f, 1.0f
};

// world space positions of our cubes
glm::vec3 cubePositions[] = {
    glm::vec3(2.0f,  5.0f, -15.0f),
    glm::vec3(-1.5f, -2.2f, -2.5f),
    glm::vec3(-3.8f, -2.0f, -12.3f),
    glm::vec3(2.4f, -0.4f, -3.5f),
    glm::vec3(-1.7f,  3.0f, -7.5f),
    glm::vec3(1.3f, -2.0f, -2.5f),
    glm::vec3(1.5f,  2.0f, -2.5f),
    glm::vec3(1.5f,  0.2f, -1.5f),
    glm::vec3(-1.3f,  1.0f, -1.5f)
};

//============================================================================================
// OpenGL state and functions.

constexpr float DarkSlateGray[] = {0.184313729f, 0.309803933f, 0.309803933f, 1.0f};

#ifdef XR_USE_PLATFORM_WIN32
    XrGraphicsBindingOpenGLWin32KHR g_graphicsBinding{XR_TYPE_GRAPHICS_BINDING_OPENGL_WIN32_KHR};
#elif defined(XR_USE_PLATFORM_XLIB)
    XrGraphicsBindingOpenGLXlibKHR g_graphicsBinding{XR_TYPE_GRAPHICS_BINDING_OPENGL_XLIB_KHR};
#elif defined(XR_USE_PLATFORM_XCB)
    XrGraphicsBindingOpenGLXcbKHR g_graphicsBinding{XR_TYPE_GRAPHICS_BINDING_OPENGL_XCB_KHR};
#elif defined(XR_USE_PLATFORM_WAYLAND)
    XrGraphicsBindingOpenGLWaylandKHR g_graphicsBinding{XR_TYPE_GRAPHICS_BINDING_OPENGL_WAYLAND_KHR};
#endif

ksGpuWindow g_window{};

std::list<std::vector<XrSwapchainImageOpenGLKHR>> g_swapchainImageBuffers;
GLuint g_swapchainFramebuffer{0};
GLuint g_program{0};
GLuint g_programSkybox{0};
GLuint g_programQuad{ 0 };
GLuint g_programScene1{ 0 };
GLint g_modelViewProjectionUniformLocation{0};
GLint g_vertexAttribCoords{0};
GLint g_vertexAttribColor{0};
GLuint g_vao{0};
GLuint g_cubeVertexBuffer{0};
GLuint g_cubeIndexBuffer{0};
GLuint skyboxVAO{0};
GLuint skyboxVBO{0};
unsigned int cubemapTexture{0};
GLuint q_vbo;
GLuint q_vao;
GLuint q_ebo;
unsigned int qudaTexture{ 0 };
GLint viewport[4];
GLuint _PBO[2];
GLuint _PBODepth[2];
GLuint pboImageBack{0};
GLuint textureBack{0};
int index = 0;
int uploadIndex{0};
GLuint pbo_read[2];
cv::Mat inpainted;

GLuint Scene1_vbo{0};
GLuint Scene1_vao{0};
unsigned int textureScene1;

std::string GetTimestamp() {
    std::time_t t = std::time(nullptr);
    char buffer[20];
    std::strftime(buffer, sizeof(buffer), "%Y%m%d%H%M%S", std::localtime(&t));
    return std::string(buffer);
}


// Map color buffer to associated depth buffer. This map is populated on demand.
std::map<uint32_t, uint32_t> g_colorToDepthMap;

std::string ReadShaderFromFile(std::string path) 
{
    std::ifstream ifs(path);
    std::string content((std::istreambuf_iterator<char>(ifs)),
        (std::istreambuf_iterator<char>()));
    return content;
}

void CheckShader(GLuint shader) {
    GLint r = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &r);
    if (r == GL_FALSE) {
        GLchar msg[4096] = {};
        GLsizei length;
        glGetShaderInfoLog(shader, sizeof(msg), &length, msg);
        THROW(Fmt("Compile shader failed: %s", msg));
    }
}

void CheckProgram(GLuint prog) {
    GLint r = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &r);
    if (r == GL_FALSE) {
        GLchar msg[4096] = {};
        GLsizei length;
        glGetProgramInfoLog(prog, sizeof(msg), &length, msg);
        THROW(Fmt("Link program failed: %s", msg));
    }
}

// loads a cubemap texture from 6 individual texture faces
// order:
// +X (right)
// -X (left)
// +Y (top)
// -Y (bottom)
// +Z (front) 
// -Z (back)
// -------------------------------------------------------
unsigned int loadCubemap(std::vector<std::string> faces)
{
    unsigned int textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_CUBE_MAP, textureID);

    int width, height, nrChannels;
    for (unsigned int i = 0; i < faces.size(); i++)
    {
        unsigned char* data = stbi_load(faces[i].c_str(), &width, &height, &nrChannels, 0);
        if (data)
        {
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
            stbi_image_free(data);
        }
        else
        {
            std::cout << "Cubemap texture failed to load at path: " << faces[i] << std::endl;
            stbi_image_free(data);
        }
    }
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

    return textureID;
}

unsigned int loadquadTexture(std::string path)
{
    // load and create a texture 
    // -------------------------
    unsigned int texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture); // all upcoming GL_TEXTURE_2D operations now have effect on this texture object
    // set the texture wrapping parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	// set texture wrapping to GL_REPEAT (default wrapping method)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    // set texture filtering parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // load image, create texture and generate mipmaps
    int width, height, nrChannels;
    // The FileSystem::getPath(...) is part of the GitHub repository so we can find files on any IDE/platform; replace it with your own image path.
    unsigned char* data = stbi_load(path.c_str(), &width, &height, &nrChannels, 0);
    if (data)
    {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
    }
    else
    {
        std::cout << "Failed to load texture" << std::endl;
    }
    stbi_image_free(data);
    return texture;
}


unsigned int loadInpaintedTexture(cv::Mat& image)
{
    // load and create a texture 
    // -------------------------
    unsigned int texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture); // all upcoming GL_TEXTURE_2D operations now have effect on this texture object
    // set the texture wrapping parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	// set texture wrapping to GL_REPEAT (default wrapping method)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    // set texture filtering parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // load image, create texture and generate mipmaps
    int width, height, nrChannels;
    // The FileSystem::getPath(...) is part of the GitHub repository so we can find files on any IDE/platform; replace it with your own image path.
    //unsigned char* data = stbi_load(path.c_str(), &width, &height, &nrChannels, 0);
    //if (data)
    //{
    //    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
    //    glGenerateMipmap(GL_TEXTURE_2D);
    //}
    //else
    //{
    //    std::cout << "Failed to load texture" << std::endl;
    //}
    //stbi_image_free(data);
    if(image.data)
    {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.cols, image.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, image.data);
        glGenerateMipmap(GL_TEXTURE_2D);
    }
    else 
    {
        std::cout << "Failed to load texture" << std::endl;
        return 0;
    }
    return texture;
}

std::unique_ptr<Model> ourModelPtr;
static void OpenGLInitializeResources()
{
    glGenFramebuffers(1, &g_swapchainFramebuffer);

    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    std::string vertexShaderStr = ReadShaderFromFile("basic.vert");
    const char* c_str = vertexShaderStr.c_str();
    glShaderSource(vertexShader, 1, &c_str, nullptr);
    glCompileShader(vertexShader);
    CheckShader(vertexShader);

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    std::string fragmentShaderStr = ReadShaderFromFile("basic.frag");
    c_str = fragmentShaderStr.c_str();
    glShaderSource(fragmentShader, 1, &c_str, nullptr);
    glCompileShader(fragmentShader);
    CheckShader(fragmentShader);

    g_program = glCreateProgram();
    glAttachShader(g_program, vertexShader);
    glAttachShader(g_program, fragmentShader);
    glLinkProgram(g_program);
    CheckProgram(g_program);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    g_modelViewProjectionUniformLocation = glGetUniformLocation(g_program, "ModelViewProjection");

    g_vertexAttribCoords = glGetAttribLocation(g_program, "VertexPos");
    g_vertexAttribColor = glGetAttribLocation(g_program, "VertexColor");

    // skybox shader
    GLuint vertexShader1 = glCreateShader(GL_VERTEX_SHADER);
    std::string vertexShaderStr1 = ReadShaderFromFile("skybox.vert");
    const char* c_str1 = vertexShaderStr1.c_str();
    glShaderSource(vertexShader1, 1, &c_str1, nullptr);
    glCompileShader(vertexShader1);
    CheckShader(vertexShader1);

    GLuint fragmentShader1 = glCreateShader(GL_FRAGMENT_SHADER);
    std::string fragmentShaderStr1 = ReadShaderFromFile("skybox.frag");
    c_str1 = fragmentShaderStr1.c_str();
    glShaderSource(fragmentShader1, 1, &c_str1, nullptr);
    glCompileShader(fragmentShader1);
    CheckShader(fragmentShader1);

    g_programSkybox = glCreateProgram();
    glAttachShader(g_programSkybox, vertexShader1);
    glAttachShader(g_programSkybox, fragmentShader1);
    glLinkProgram(g_programSkybox);
    CheckProgram(g_programSkybox);

    // delete the shaders as they're linked into our program now and no longer necessary
    glDeleteShader(vertexShader1);
    glDeleteShader(fragmentShader1);

    //qudaShader(plane for right eye)
    GLuint vertexShader2 = glCreateShader(GL_VERTEX_SHADER);
    std::string vertexShaderStr2 = ReadShaderFromFile("quad.vert");
    const char* c_str2 = vertexShaderStr2.c_str();
    glShaderSource(vertexShader2, 1, &c_str2, nullptr);
    glCompileShader(vertexShader2);
    CheckShader(vertexShader2);

    GLuint fragmentShader2 = glCreateShader(GL_FRAGMENT_SHADER);
    std::string fragmentShaderStr2 = ReadShaderFromFile("quad.frag");
    c_str2 = fragmentShaderStr2.c_str();
    glShaderSource(fragmentShader2, 1, &c_str2, nullptr);
    glCompileShader(fragmentShader2);
    CheckShader(fragmentShader2);

    g_programQuad = glCreateProgram(); 
    glAttachShader(g_programQuad, vertexShader2);
    glAttachShader(g_programQuad, fragmentShader2);
    glLinkProgram(g_programQuad);
    CheckProgram(g_programQuad);

    //// delete the shaders as they're linked into our program now and no longer necessary
    glDeleteShader(vertexShader2);
    glDeleteShader(fragmentShader2);

    // scene shader
    GLuint vertexScene1Shader = glCreateShader(GL_VERTEX_SHADER);
    std::cout << "problem here" << std::endl;
    std::string vertexScene1ShaderStr = ReadShaderFromFile("scene1.vert");
    const char* c_strs1 = vertexScene1ShaderStr.c_str();
    glShaderSource(vertexScene1Shader, 1, &c_strs1, nullptr);
    glCompileShader(vertexScene1Shader);
    CheckShader(vertexScene1Shader);

    GLuint fragmentScene1Shader = glCreateShader(GL_FRAGMENT_SHADER);
    std::string fragmentScene1ShaderStr = ReadShaderFromFile("scene1.frag");
    c_strs1 = fragmentScene1ShaderStr.c_str();
    glShaderSource(fragmentScene1Shader, 1, &c_strs1, nullptr);
    glCompileShader(fragmentScene1Shader);
    CheckShader(fragmentScene1Shader);

    g_programScene1 = glCreateProgram(); 
    glAttachShader(g_programScene1, vertexScene1Shader);
    glAttachShader(g_programScene1, fragmentScene1Shader);
    glLinkProgram(g_programScene1);
    CheckProgram(g_programScene1);

    ////// delete the shaders as they're linked into our program now and no longer necessary
    glDeleteShader(vertexScene1Shader);
    glDeleteShader(fragmentScene1Shader);


    //SKYBOX
    //unsigned int skyboxVAO, skyboxVBO;
    glGenVertexArrays(1, &skyboxVAO);
    glGenBuffers(1, &skyboxVBO);
    glBindVertexArray(skyboxVAO);
    glBindBuffer(GL_ARRAY_BUFFER, skyboxVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(skyboxVertices), &skyboxVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);


    // bind vbo
    glGenBuffers(1, &g_cubeVertexBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, g_cubeVertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(Geometry::c_cubeVertices), Geometry::c_cubeVertices, GL_STATIC_DRAW);

    glGenBuffers(1, &g_cubeIndexBuffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_cubeIndexBuffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(Geometry::c_cubeIndices), Geometry::c_cubeIndices, GL_STATIC_DRAW);

    glGenVertexArrays(1, &g_vao);
    glBindVertexArray(g_vao);
    glEnableVertexAttribArray(g_vertexAttribCoords);
    glEnableVertexAttribArray(g_vertexAttribColor);
    glBindBuffer(GL_ARRAY_BUFFER, g_cubeVertexBuffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_cubeIndexBuffer);
    glVertexAttribPointer(g_vertexAttribCoords, 3, GL_FLOAT, GL_FALSE, sizeof(Geometry::Vertex), nullptr);
    glVertexAttribPointer(g_vertexAttribColor, 3, GL_FLOAT, GL_FALSE, sizeof(Geometry::Vertex),
                          reinterpret_cast<const void*>(sizeof(XrVector3f)));


    std::vector<std::string> faces
    {
        "F:/VR/OpenXR/OpenXR-OpenGL-Example/skybox/right.jpg",
        "F:/VR/OpenXR/OpenXR-OpenGL-Example/skybox/left.jpg",
        "F:/VR/OpenXR/OpenXR-OpenGL-Example/skybox/top.jpg",
        "F:/VR/OpenXR/OpenXR-OpenGL-Example/skybox/bottom.jpg",
        "F:/VR/OpenXR/OpenXR-OpenGL-Example/skybox/front.jpg",
        "F:/VR/OpenXR/OpenXR-OpenGL-Example/skybox/back.jpg"
    };
    cubemapTexture = loadCubemap(faces);

    glUseProgram(g_programSkybox);
    glUniform1i(glGetUniformLocation(g_programSkybox, "skybox"), 0);

    // quad(plane for right eye)
    //unsigned int VBO, VAO, EBO;
    glGenVertexArrays(1, &q_vao);
    glGenBuffers(1, &q_vbo);
    glGenBuffers(1, &q_ebo);

    glBindVertexArray(q_vao);

    glBindBuffer(GL_ARRAY_BUFFER, q_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, q_ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // color attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    // texture coord attribute
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);
    std::string path = "F:/VR/OpenXR/OpenXR-OpenGL-Example/skybox/test/container.jpg";
    qudaTexture = loadquadTexture(path);
    
    
    // for the pbo of framebuffer initilization
    GLuint Width = 2064;
    GLuint Height = 2096;
    
    
    glGenBuffers(2, _PBO);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, _PBO[0]);
    glBufferData(GL_PIXEL_PACK_BUFFER, Width* Height * 4, 0, GL_STREAM_READ);

    glBindBuffer(GL_PIXEL_PACK_BUFFER, _PBO[1]);
    glBufferData(GL_PIXEL_PACK_BUFFER, Width* Height * 4, 0, GL_STREAM_READ);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

    // for the pbo of depth buffer initilization
    glGenBuffers(2, _PBODepth);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, _PBODepth[0]);
    glBufferData(GL_PIXEL_PACK_BUFFER, Width* Height * 4, 0, GL_STREAM_READ);

    glBindBuffer(GL_PIXEL_PACK_BUFFER, _PBODepth[1]);
    glBufferData(GL_PIXEL_PACK_BUFFER, Width* Height * 4, 0, GL_STREAM_READ);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
  
    //for pbo read back to right eye plane
    glGenBuffers(2, pbo_read);
    for (int i = 0; i < 2; i++) {
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_read[i]);

        // Initialize with enough storage capacity (e.g. for an image of width x height x 3 bytes)
        glBufferData(GL_PIXEL_UNPACK_BUFFER, Width * Height * 4, nullptr, GL_STREAM_DRAW);

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);  // Unbind the PBO for now
    }


    glGenTextures(1, &textureBack);
    glBindTexture(GL_TEXTURE_2D, textureBack);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, Width, Height, 0, GL_BGR, GL_UNSIGNED_BYTE, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    
    // initilzation the model
    ourModelPtr = std::make_unique<Model>("F:/VR/OpenXR/OpenXR-OpenGL-model/skybox/objects/backpack/backpack.obj");
}


static void OpenGLInitializeDevice(XrInstance instance, XrSystemId systemId)
{
    // Extension function must be loaded by name
    PFN_xrGetOpenGLGraphicsRequirementsKHR pfnGetOpenGLGraphicsRequirementsKHR = nullptr;
    CHECK_XRCMD(xrGetInstanceProcAddr(instance, "xrGetOpenGLGraphicsRequirementsKHR",
                                      reinterpret_cast<PFN_xrVoidFunction*>(&pfnGetOpenGLGraphicsRequirementsKHR)));

    XrGraphicsRequirementsOpenGLKHR graphicsRequirements{XR_TYPE_GRAPHICS_REQUIREMENTS_OPENGL_KHR};
    CHECK_XRCMD(pfnGetOpenGLGraphicsRequirementsKHR(instance, systemId, &graphicsRequirements));

    // Initialize the gl extensions. Note we have to open a window.
    ksDriverInstance driverInstance{};
    ksGpuQueueInfo queueInfo{};
    ksGpuSurfaceColorFormat colorFormat{KS_GPU_SURFACE_COLOR_FORMAT_B8G8R8A8};
    ksGpuSurfaceDepthFormat depthFormat{KS_GPU_SURFACE_DEPTH_FORMAT_D24};
    ksGpuSampleCount sampleCount{KS_GPU_SAMPLE_COUNT_1};
    if (!ksGpuWindow_Create(&g_window, &driverInstance, &queueInfo, 0, colorFormat, depthFormat, sampleCount, 640, 480, false)) {
        THROW("Unable to create GL context");
    }

    GLint major = 0;
    GLint minor = 0;
    glGetIntegerv(GL_MAJOR_VERSION, &major);
    glGetIntegerv(GL_MINOR_VERSION, &minor);

    const XrVersion desiredApiVersion = XR_MAKE_VERSION(major, minor, 0);
    if (graphicsRequirements.minApiVersionSupported > desiredApiVersion) {
        THROW("Runtime does not support desired Graphics API and/or version");
    }
#ifdef XR_USE_PLATFORM_WIN32  
    g_graphicsBinding.hDC = g_window.context.hDC;
    g_graphicsBinding.hGLRC = g_window.context.hGLRC;
#elif defined(XR_USE_PLATFORM_XLIB)
    g_graphicsBinding.xDisplay = g_window.context.xDisplay;
    g_graphicsBinding.visualid = g_window.context.visualid;
    g_graphicsBinding.glxFBConfig = g_window.context.glxFBConfig;
    g_graphicsBinding.glxDrawable = g_window.context.glxDrawable;
    g_graphicsBinding.glxContext = g_window.context.glxContext;
#elif defined(XR_USE_PLATFORM_XCB)
    // TODO: Still missing the platform adapter, and some items to make this usable.
    g_graphicsBinding.connection = g_window.connection;
    // g_graphicsBinding.screenNumber = g_window.context.screenNumber;
    // g_graphicsBinding.fbconfigid = g_window.context.fbconfigid;
    g_graphicsBinding.visualid = g_window.context.visualid;
    g_graphicsBinding.glxDrawable = g_window.context.glxDrawable;
    // g_graphicsBinding.glxContext = g_window.context.glxContext;
#elif defined(XR_USE_PLATFORM_WAYLAND)
    // TODO: Just need something other than NULL here for now (for validation).  Eventually need
    //       to correctly put in a valid pointer to an wl_display
    g_graphicsBinding.display = reinterpret_cast<wl_display*>(0xFFFFFFFF);
#endif

    glEnable(GL_DEBUG_OUTPUT);
    glDebugMessageCallback(
        [](GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message,
           const void* userParam) {
            std::cout << "GL Debug: " << std::string(message, 0, length) << std::endl;
        },
        nullptr);

    OpenGLInitializeResources();
}

static int64_t OpenGLSelectColorSwapchainFormat(const std::vector<int64_t>& runtimeFormats)
{
    // List of supported color swapchain formats.
    constexpr int64_t SupportedColorSwapchainFormats[] = {
        GL_RGB10_A2,
        GL_RGBA16F,
        // The two below should only be used as a fallback, as they are linear color formats without enough bits for color
        // depth, thus leading to banding.
        GL_RGBA8,
        GL_RGBA8_SNORM,
    };

    auto swapchainFormatIt =
        std::find_first_of(runtimeFormats.begin(), runtimeFormats.end(), std::begin(SupportedColorSwapchainFormats),
                           std::end(SupportedColorSwapchainFormats));
    if (swapchainFormatIt == runtimeFormats.end()) {
        THROW("No runtime swapchain format supported for color swapchain");
    }

    return *swapchainFormatIt;
}

static std::vector<XrSwapchainImageBaseHeader*> OpenGLAllocateSwapchainImageStructs(
        uint32_t capacity, const XrSwapchainCreateInfo& /*swapchainCreateInfo*/)
{
    // Allocate and initialize the buffer of image structs (must be sequential in memory for xrEnumerateSwapchainImages).
    // Return back an array of pointers to each swapchain image struct so the consumer doesn't need to know the type/size.
    std::vector<XrSwapchainImageOpenGLKHR> swapchainImageBuffer(capacity);
    std::vector<XrSwapchainImageBaseHeader*> swapchainImageBase;
    for (XrSwapchainImageOpenGLKHR& image : swapchainImageBuffer) {
        image.type = XR_TYPE_SWAPCHAIN_IMAGE_OPENGL_KHR;
        swapchainImageBase.push_back(reinterpret_cast<XrSwapchainImageBaseHeader*>(&image));
    }

    // Keep the buffer alive by moving it into the list of buffers.
    g_swapchainImageBuffers.push_back(std::move(swapchainImageBuffer));

    return swapchainImageBase;
}

static uint32_t OpenGLGetDepthTexture(uint32_t colorTexture)
{
    // If a depth-stencil view has already been created for this back-buffer, use it.
    auto depthBufferIt = g_colorToDepthMap.find(colorTexture);
    if (depthBufferIt != g_colorToDepthMap.end()) {
        return depthBufferIt->second;
    }

    // This back-buffer has no corresponding depth-stencil texture, so create one with matching dimensions.

    GLint width;
    GLint height;
    glBindTexture(GL_TEXTURE_2D, colorTexture);
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &width);
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &height);

    uint32_t depthTexture;
    glGenTextures(1, &depthTexture);
    glBindTexture(GL_TEXTURE_2D, depthTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);

    g_colorToDepthMap.insert(std::make_pair(colorTexture, depthTexture));

    return depthTexture;
}

//static void OpenGLRenderView_Test(const XrCompositionLayerProjectionView& layerView, const XrSwapchainImageBaseHeader* swapchainImage,
//                int64_t swapchainFormat, const std::vector<Space>& spaces)
//{
//    CHECK(layerView.subImage.imageArrayIndex == 0);  // Texture arrays not supported.
//    UNUSED_PARM(swapchainFormat);                    // Not used in this function for now.
//
//    glBindFramebuffer(GL_FRAMEBUFFER, g_swapchainFramebuffer);
//
//    const uint32_t colorTexture = reinterpret_cast<const XrSwapchainImageOpenGLKHR*>(swapchainImage)->image;
//
//    glViewport(static_cast<GLint>(layerView.subImage.imageRect.offset.x),
//               static_cast<GLint>(layerView.subImage.imageRect.offset.y),
//               static_cast<GLsizei>(layerView.subImage.imageRect.extent.width),
//               static_cast<GLsizei>(layerView.subImage.imageRect.extent.height));
//
//    glFrontFace(GL_CW);
//    glCullFace(GL_BACK);
//    // Disable back-face culling so we can see the inside of the world-space cube
//    glDisable(GL_CULL_FACE);
//    //glEnable(GL_CULL_FACE);
//    glEnable(GL_DEPTH_TEST);
//    //glEnable(GL_TEXTURE_2D);
//    const uint32_t depthTexture = OpenGLGetDepthTexture(colorTexture);
//
//    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorTexture, 0);
//    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthTexture, 0);
//
//    // Clear swapchain and depth buffer.
//    glClearColor(DarkSlateGray[0], DarkSlateGray[1], DarkSlateGray[2], DarkSlateGray[3]);
//    glClearDepth(1.0f);
//    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
//
//    // Set shaders and uniform variables.
//    glUseProgram(g_program);
//    
//    const auto& pose = layerView.pose;
//    XrMatrix4x4f proj;
//    XrMatrix4x4f_CreateProjectionFov(&proj, GRAPHICS_OPENGL, layerView.fov, 0.05f, 100.0f);
//    XrMatrix4x4f toView;
//    XrVector3f scale{1.f, 1.f, 1.f};
//    XrMatrix4x4f_CreateTranslationRotationScale(&toView, &pose.position, &pose.orientation, &scale);
//    XrMatrix4x4f view;
//    XrMatrix4x4f_InvertRigidBody(&view, &toView);
//    XrMatrix4x4f vp;
//    XrMatrix4x4f_Multiply(&vp, &proj, &view);
//
//    // Set cube primitive data.
//    glBindVertexArray(g_vao);
//
//    glBindBuffer(GL_ARRAY_BUFFER, g_cubeVertexBuffer);
//    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_cubeIndexBuffer);
//    // Things drawn here will appear in world space at the scale specified above (if scale = 1 then
//    // unit scale).  The Model transform and scale will adjust where and how large they are.
//    // Here, we draw a cube that is 10 meters large located at the origin.
//    /// @todo Replace with the things you'd like to be drawn in the world.
//    {
//        XrPosef id = Math::Pose::Identity();
//        XrVector3f worldCubeScale{10.f, 3.f, 10.f};
//        // Compute the model-view-projection transform and set it..
//        XrMatrix4x4f model;
//        XrMatrix4x4f_CreateTranslationRotationScale(&model, &id.position, &id.orientation, &worldCubeScale);
//        XrMatrix4x4f mvp;
//        XrMatrix4x4f_Multiply(&mvp, &vp, &model);
//        glUniformMatrix4fv(g_modelViewProjectionUniformLocation, 1, GL_FALSE, reinterpret_cast<const GLfloat*>(&mvp));
//
//        // Draw the world cube.
//        //glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(ArraySize(Geometry::c_cubeIndices)), GL_UNSIGNED_SHORT, nullptr);
//    }
//
//    // Render a cube within each of the spaces we've been asked to render, at the requested sizes.  These show
//    // the centers of each of the spaces we defined.
//    /// @todo Use the name of each space to determine what to draw in it.
//    //int i = 0;
//    for (const Space& space : spaces) {
//        if (g_verbosity >= 10) {
//            std::cout << " Rendering " << space.Name << " space" << std::endl;
//        }
//        // Compute the model-view-projection transform and set it..
//        XrMatrix4x4f model;
//        XrMatrix4x4f_CreateTranslationRotationScale(&model, &space.Pose.position, &space.Pose.orientation, &space.Scale);
//        XrMatrix4x4f mvp;
//        XrMatrix4x4f_Multiply(&mvp, &vp, &model);
//        glUniformMatrix4fv(g_modelViewProjectionUniformLocation, 1, GL_FALSE, reinterpret_cast<const GLfloat*>(&mvp));
//
//        // Draw the cube.
//        glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(ArraySize(Geometry::c_cubeIndices)), GL_UNSIGNED_SHORT, nullptr);
//    }
//
//
//
//    {
//        // draw skybox as last
//        glDepthFunc(GL_LEQUAL);  // change depth function so depth test passes when values are equal to depth buffer's content
//        glUseProgram(g_programSkybox);
//        //view = glm::mat4(glm::mat3(camera.GetViewMatrix())); // remove translation from the view matrix
//       //skyboxShader.setMat4("view", view);
//        glUniformMatrix4fv(glGetUniformLocation(g_programSkybox, "view"), 1, GL_FALSE, reinterpret_cast<const GLfloat*>(&view));
//        //skyboxShader.setMat4("projection", projection);
//        glUniformMatrix4fv(glGetUniformLocation(g_programSkybox, "projection"), 1, GL_FALSE, reinterpret_cast<const GLfloat*>(&proj));
//        // skybox cube
//        glBindVertexArray(skyboxVAO);
//        glActiveTexture(GL_TEXTURE0);
//        glBindTexture(GL_TEXTURE_CUBE_MAP, cubemapTexture);
//        glDrawArrays(GL_TRIANGLES, 0, 36);
//        glBindVertexArray(0);
//        glDepthFunc(GL_LESS); // set depth function back to default
//    }
//
//   
//    //cv::Mat dst;
//    // Convert to grayscale
//    //cv::Mat gray;
//    //cv::cvtColor(image1, gray, cv::COLOR_BGRA2GRAY);
//    //cv::threshold(image1, dst, 254, 255, cv::THRESH_BINARY_INV);  // Convert white to black and vice versa
//    //cv::imshow("mask frame", image1); // Display the image
//    //cv::waitKey(1); // Update the window
//
//    //GLint viewport[4];
//    //glGetIntegerv(GL_VIEWPORT, viewport);
//    //int x = viewport[0];
//    //int y = viewport[1];
//    //int width = viewport[2];
//    //int height = viewport[3];
//
//    //std::vector<float> depthValues(width * height);
//    //glReadPixels(x, y, width, height, GL_DEPTH_COMPONENT, GL_FLOAT, depthValues.data());
//
//    //cv::Mat depthMap(height, width, CV_32FC1, depthValues.data());
//    //cv::flip(depthMap, depthMap, 0);
//
//    //cv::imshow("OpenGL depth", depthMap); // Display the image
//    //cv::waitKey(1); // Update the window
//    
//    // for the pbo
//// Bind the PBO for the current frame
//
//    GLint viewport[4];
//    glGetIntegerv(GL_VIEWPORT, viewport);
//    int x = viewport[0];
//    int y = viewport[1];
//    int width = viewport[2];
//    int height = viewport[3];
//    glBindBuffer(GL_PIXEL_PACK_BUFFER, _PBO[index]);
//
//    // Read the pixels into the PBO
//    glReadPixels(x, y, width, height, GL_BGR, GL_UNSIGNED_BYTE, 0);
//
//    // Process the pixels from the previous frame
//    glBindBuffer(GL_PIXEL_PACK_BUFFER, _PBO[1 - index]);
//    GLubyte* ptr = (GLubyte*)glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);
//    
//    //cv::Mat inpainted;
//    if (ptr)
//    {
//        cv::Mat img(height, width, CV_8UC3, ptr);
//        // Process img with OpenCV...
//        cv::flip(img, img, 0); // Flip the image vertically
//
//        //cv::imshow("OpenGL Frame", img); // Display the image
//        //cv::waitKey(1); // Update the window
//
//        cv::Mat gray;
//        cv::cvtColor(img, gray, cv::COLOR_BGRA2GRAY);
//
//        // Threshold the grayscale image
//        cv::Mat binary;
//        cv::threshold(gray, binary, 254, 255, cv::THRESH_BINARY_INV);
//        //cv::imshow("OpenGL Frame", binary); // Display the image
//        //cv::waitKey(1); // Update the window
//        dr::PixMix pm;
//        dr::det::PixMixParams params;
//        params.alpha = 0.0f;			// 0.0f means no spatial cost considered
//        params.maxItr = 1;				// set to 1 to crank up the speed
//        params.maxRandSearchItr = 1;	// set to 1 to crank up the speed
//        bool debugViz = false;
//        pm.Run(img, binary, inpainted, params, debugViz);
//        //inpainted = img.clone();
//        cv::imshow("Inpainted color image", inpainted);
//        cv::waitKey(1);;
//        // Unmap the buffer
//        glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
//    }
//    
//    // Unbind the PBO
//    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
//
//    // flip index for next frame
//    index = 1 - index;
//
//    
//    // draw a plane
//    //cv::Mat image1 = CaptureOpenGLFrame();
//
//
//    ////cv::imshow("Inpainted color image1", image1);
//    ////cv::waitKey(1);
//
//    // transfer back to OpengL
//    //cv::flip(inpainted, inpainted, 0);
//    //glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_read[uploadIndex]);
//    //GLubyte* mem = (GLubyte*)glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY);
//    //if (mem)
//    //{
//    //    memcpy(mem, inpainted.data, width * height * 3);  // Assumes the image is BGR
//    //    glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);
//
//    //    // Update texture with the PBO data
//    //    glBindTexture(GL_TEXTURE_2D, textureBack);
//    //    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_BGR, GL_UNSIGNED_BYTE, 0);
//    //    glUseProgram(g_programQuad);
//    //    glBindVertexArray(q_vao);
//    //    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
//
//
//    //}
//
//    //glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
//
//    //uploadIndex = 1 - uploadIndex;
//
//
//
//
//    // single pbo
//    //glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboImageBack);
//    //void* mem = glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY);
//    //memcpy(mem, inpainted.ptr(), width* height* 3); // Copy image data into PBO
//    //glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);
//
//    //// Use PBO to update texture
//    //glBindTexture(GL_TEXTURE_2D, textureBack);
//    //glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboImageBack);
//    //glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_BGR, GL_UNSIGNED_BYTE, 0);
//    //glUseProgram(g_programQuad);
//    //glBindVertexArray(q_vao);
//    //glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
//    //glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
//    
//    // regular 
//    //qudaTexture = loadInpaintedTexture(inpainted);
//    //{
//    //    glBindTexture(GL_TEXTURE_2D, qudaTexture);
//
//    //    // render container
//    //    glUseProgram(g_programQuad);
//    //    glBindVertexArray(q_vao);
//    //    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
//    //}
//
//    //glDeleteTextures(1, &qudaTexture);
//
//    //glDeleteTextures(1, &textureBack);
//    //glDeleteBuffers(1, &pboImageBack);
//    glBindVertexArray(0);
//    glUseProgram(0);
//
//    glBindFramebuffer(GL_FRAMEBUFFER, 0);
//
//    // Swap our window every other eye for RenderDoc
//    static int everyOther = 0;
//    if ((everyOther++ & 1) != 0) {
//        ksGpuWindow_SwapBuffers(&g_window);
//    }
//}


static void OpenGLRenderViewRight(const XrCompositionLayerProjectionView& layerView, const XrSwapchainImageBaseHeader* swapchainImage,
    int64_t swapchainFormat, const std::vector<Space>& spaces)
{
    CHECK(layerView.subImage.imageArrayIndex == 0);  // Texture arrays not supported.
    UNUSED_PARM(swapchainFormat);                    // Not used in this function for now.

    glBindFramebuffer(GL_FRAMEBUFFER, g_swapchainFramebuffer);

    const uint32_t colorTexture = reinterpret_cast<const XrSwapchainImageOpenGLKHR*>(swapchainImage)->image;

    glViewport(static_cast<GLint>(layerView.subImage.imageRect.offset.x),
        static_cast<GLint>(layerView.subImage.imageRect.offset.y),
        static_cast<GLsizei>(layerView.subImage.imageRect.extent.width),
        static_cast<GLsizei>(layerView.subImage.imageRect.extent.height));

    glFrontFace(GL_CW);
    glCullFace(GL_BACK);
    // Disable back-face culling so we can see the inside of the world-space cube
    glDisable(GL_CULL_FACE);
    //glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);

    const uint32_t depthTexture = OpenGLGetDepthTexture(colorTexture);

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorTexture, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthTexture, 0);

    // Clear swapchain and depth buffer.
    glClearColor(DarkSlateGray[0], DarkSlateGray[1], DarkSlateGray[2], DarkSlateGray[3]);
    glClearDepth(1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    // Set shaders and uniform variables.
    glUseProgram(g_program);

    const auto& pose = layerView.pose;
    XrMatrix4x4f proj;
    XrMatrix4x4f_CreateProjectionFov(&proj, GRAPHICS_OPENGL, layerView.fov, 0.05f, 100.0f);
    XrMatrix4x4f toView;
    XrVector3f scale{ 1.f, 1.f, 1.f };
    XrMatrix4x4f_CreateTranslationRotationScale(&toView, &pose.position, &pose.orientation, &scale);
    XrMatrix4x4f view;
    XrMatrix4x4f_InvertRigidBody(&view, &toView);
    XrMatrix4x4f vp;
    XrMatrix4x4f_Multiply(&vp, &proj, &view);


    // render the inpainted image back to right eye
    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);
    int x = viewport[0];
    int y = viewport[1];
    int width = viewport[2];
    int height = viewport[3];

    cv::flip(inpainted, inpainted, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_read[uploadIndex]);
    GLubyte* mem = (GLubyte*)glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY);
    if (mem)
    {
        memcpy(mem, inpainted.data, width * height * 3);  // Assumes the image is BGR
        glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);

        // Update texture with the PBO data
        glBindTexture(GL_TEXTURE_2D, textureBack);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_BGR, GL_UNSIGNED_BYTE, 0);
        glUseProgram(g_programQuad);
        glBindVertexArray(q_vao);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);


    }

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    uploadIndex = 1 - uploadIndex;

    glBindVertexArray(0);
    glUseProgram(0);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // Swap our window every other eye for RenderDoc
    static int everyOther = 0;
    if ((everyOther++ & 1) != 0) {
        ksGpuWindow_SwapBuffers(&g_window);
    }

    cv::flip(inpainted, inpainted, 0);
}


static void OpenGLRenderViewScene1(const XrCompositionLayerProjectionView& layerView, const XrSwapchainImageBaseHeader* swapchainImage,
    int64_t swapchainFormat, const std::vector<Space>& spaces)
{
    CHECK(layerView.subImage.imageArrayIndex == 0);  // Texture arrays not supported.
    UNUSED_PARM(swapchainFormat);                    // Not used in this function for now.

    glBindFramebuffer(GL_FRAMEBUFFER, g_swapchainFramebuffer);

    const uint32_t colorTexture = reinterpret_cast<const XrSwapchainImageOpenGLKHR*>(swapchainImage)->image;

    glViewport(static_cast<GLint>(layerView.subImage.imageRect.offset.x),
        static_cast<GLint>(layerView.subImage.imageRect.offset.y),
        static_cast<GLsizei>(layerView.subImage.imageRect.extent.width),
        static_cast<GLsizei>(layerView.subImage.imageRect.extent.height));

    glFrontFace(GL_CW);
    glCullFace(GL_BACK);
    // Disable back-face culling so we can see the inside of the world-space cube
    glDisable(GL_CULL_FACE);
    //glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    //glEnable(GL_TEXTURE_2D);
    const uint32_t depthTexture = OpenGLGetDepthTexture(colorTexture);

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorTexture, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthTexture, 0);

    // Clear swapchain and depth buffer.
    glClearColor(DarkSlateGray[0], DarkSlateGray[1], DarkSlateGray[2], DarkSlateGray[3]);
    glClearDepth(1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    // Set shaders and uniform variables.
    glUseProgram(g_program);

    const auto& pose = layerView.pose;
    XrMatrix4x4f proj;
    XrMatrix4x4f_CreateProjectionFov(&proj, GRAPHICS_OPENGL, layerView.fov, 0.05f, 100.0f);
    XrMatrix4x4f toView;
    XrVector3f scale{ 1.f, 1.f, 1.f };
    XrMatrix4x4f_CreateTranslationRotationScale(&toView, &pose.position, &pose.orientation, &scale);
    XrMatrix4x4f view;
    XrMatrix4x4f_InvertRigidBody(&view, &toView);
    XrMatrix4x4f vp;
    XrMatrix4x4f_Multiply(&vp, &proj, &view);

    // Set cube primitive data.
    glBindVertexArray(g_vao);

    glBindBuffer(GL_ARRAY_BUFFER, g_cubeVertexBuffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_cubeIndexBuffer);

    // Render a cube within each of the spaces we've been asked to render, at the requested sizes.  These show
    // the centers of each of the spaces we defined.
    /// @todo Use the name of each space to determine what to draw in it.
    //int i = 0;
    for (const Space& space : spaces) {
        if (g_verbosity >= 10) {
            std::cout << " Rendering " << space.Name << " space" << std::endl;
        }
        // Compute the model-view-projection transform and set it..
        XrMatrix4x4f model;
        XrMatrix4x4f_CreateTranslationRotationScale(&model, &space.Pose.position, &space.Pose.orientation, &space.Scale);
        XrMatrix4x4f mvp;
        XrMatrix4x4f_Multiply(&mvp, &vp, &model);
        glUniformMatrix4fv(g_modelViewProjectionUniformLocation, 1, GL_FALSE, reinterpret_cast<const GLfloat*>(&mvp));

        // Draw the cube.
        glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(ArraySize(Geometry::c_cubeIndices)), GL_UNSIGNED_SHORT, nullptr);
    }



    {
        // draw skybox as last
        glDepthFunc(GL_LEQUAL);  // change depth function so depth test passes when values are equal to depth buffer's content
        glUseProgram(g_programSkybox);
        //view = glm::mat4(glm::mat3(camera.GetViewMatrix())); // remove translation from the view matrix
       //skyboxShader.setMat4("view", view);
        glUniformMatrix4fv(glGetUniformLocation(g_programSkybox, "view"), 1, GL_FALSE, reinterpret_cast<const GLfloat*>(&view));
        //skyboxShader.setMat4("projection", projection);
        glUniformMatrix4fv(glGetUniformLocation(g_programSkybox, "projection"), 1, GL_FALSE, reinterpret_cast<const GLfloat*>(&proj));
        // skybox cube
        glBindVertexArray(skyboxVAO);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_CUBE_MAP, cubemapTexture);
        glDrawArrays(GL_TRIANGLES, 0, 36);
        glBindVertexArray(0);
        glDepthFunc(GL_LESS); // set depth function back to default
    }

    glUseProgram(g_programScene1);
    glUniformMatrix4fv(glGetUniformLocation(g_programScene1, "projection"), 1, GL_FALSE, reinterpret_cast<const GLfloat*>(&proj));
    glUniformMatrix4fv(glGetUniformLocation(g_programScene1, "view"), 1, GL_FALSE, reinterpret_cast<const GLfloat*>(&view));

    // render the loaded model
    glm::mat4 model = glm::mat4(1.0f);
    model = glm::translate(model, glm::vec3(0.0f, 0.0f, -1.0f)); // translate it down so it's at the center of the scene
    model = glm::scale(model, glm::vec3(0.5f, 0.5f, 0.5f));	// it's a bit too big for our scene, so scale it down
    //ourShader.setMat4("model", model);
    glUniformMatrix4fv(glGetUniformLocation(g_programScene1, "model"), 1, GL_FALSE, reinterpret_cast<const GLfloat*>(&model));
    //ourModel.Draw(ourShader);
    ourModelPtr->Draw(g_programScene1);

    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);
    int x = viewport[0];
    int y = viewport[1];
    int width = viewport[2];
    int height = viewport[3];

    
    // Read the color framebuffer
    glBindBuffer(GL_PIXEL_PACK_BUFFER, _PBO[index]);
    glReadPixels(x, y, width, height, GL_BGR, GL_UNSIGNED_BYTE, 0);

    // Map the color data from the previous frame's PBO
    glBindBuffer(GL_PIXEL_PACK_BUFFER, _PBO[1 - index]);
    GLubyte* ptr = (GLubyte*)glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);

    // Read the depth buffer
    glBindBuffer(GL_PIXEL_PACK_BUFFER, _PBODepth[index]);
    glReadPixels(x, y, width, height, GL_DEPTH_COMPONENT, GL_FLOAT, 0);

    // Map the depth data from the previous frame's PBO
    glBindBuffer(GL_PIXEL_PACK_BUFFER, _PBODepth[1 - index]);
    float* depthPtr = (float*)glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);
   

    //cv::Mat inpainted;
    if (ptr&&depthPtr)
    {
        cv::Mat img(height, width, CV_8UC3, ptr);
        // Process img with OpenCV...
        cv::flip(img, img, 0); // Flip the image vertically
        //cv::imshow("OpenGL Frame", img); // Display the image
        //cv::waitKey(1); // Update the window

        cv::Mat depthMat(height, width, CV_32FC1, depthPtr);
        cv::Mat depthNormalized;
        cv::normalize(depthMat, depthNormalized, 0, 255, cv::NORM_MINMAX, CV_8U);
        depthNormalized = 255 - depthNormalized;
        cv::flip(depthNormalized, depthNormalized, 0);
        //cv::imshow("OpenGL Frame", depthNormalized); // Display the image
        //cv::waitKey(1); // Update the window
        cv::Mat resized_image;
        cv::resize(img, resized_image, cv::Size(img.cols / 3, img.rows / 3));
        cv::Mat resized_depthimage;
        cv::resize(depthNormalized, resized_depthimage, cv::Size(depthNormalized.cols / 3, depthNormalized.rows / 3));

        cv::Mat shift_image;
        if(mode) // image warping
        {
            shift_image = warp_image_based_on_depth(resized_image, resized_depthimage, 50);
        }
        else // add manully
        {
            img.copyTo(shift_image);
        }
        
        // create the mask image
        cv::Mat gray;
        cv::cvtColor(shift_image, gray, cv::COLOR_BGRA2GRAY);

        // Threshold the grayscale image
        cv::Mat binary;
        cv::threshold(gray, binary, 254, 255, cv::THRESH_BINARY_INV); // create mask image
        
        dr::PixMix pm;
        dr::det::PixMixParams params;
        params.alpha = 1.0f;			// 0.0f means no spatial cost considered
        params.maxItr = 3;				// set to 1 to crank up the speed
        params.maxRandSearchItr = 3;	// set to 1 to crank up the speed
        bool debugViz = false;
        pm.Execute(shift_image, binary, inpainted, params, debugViz);
       
        glBindBuffer(GL_PIXEL_PACK_BUFFER, _PBO[1 - index]);
        glUnmapBuffer(GL_PIXEL_PACK_BUFFER);

        // Unmap the depth PBO
        glBindBuffer(GL_PIXEL_PACK_BUFFER, _PBODepth[1 - index]);
        glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
    }

    // Unbind the PBO
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);


    // flip index for next frame
    index = 1 - index;

    glBindVertexArray(0);
    glUseProgram(0);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // Swap our window every other eye for RenderDoc
    static int everyOther = 0;
    if ((everyOther++ & 1) != 0) {
        ksGpuWindow_SwapBuffers(&g_window);
    }
}

//static void OpenGLRenderViewScene2_test(const XrCompositionLayerProjectionView& layerView, const XrSwapchainImageBaseHeader* swapchainImage,
//    int64_t swapchainFormat, const std::vector<Space>& spaces)
//{
//    CHECK(layerView.subImage.imageArrayIndex == 0);  // Texture arrays not supported.
//    UNUSED_PARM(swapchainFormat);                    // Not used in this function for now.
//
//    glBindFramebuffer(GL_FRAMEBUFFER, g_swapchainFramebuffer);
//
//    const uint32_t colorTexture = reinterpret_cast<const XrSwapchainImageOpenGLKHR*>(swapchainImage)->image;
//
//    glViewport(static_cast<GLint>(layerView.subImage.imageRect.offset.x),
//        static_cast<GLint>(layerView.subImage.imageRect.offset.y),
//        static_cast<GLsizei>(layerView.subImage.imageRect.extent.width),
//        static_cast<GLsizei>(layerView.subImage.imageRect.extent.height));
//
//    glFrontFace(GL_CW);
//    glCullFace(GL_BACK);
//    // Disable back-face culling so we can see the inside of the world-space cube
//    glDisable(GL_CULL_FACE);
//    //glEnable(GL_CULL_FACE);
//    glEnable(GL_DEPTH_TEST);
//    //glEnable(GL_TEXTURE_2D);
//    const uint32_t depthTexture = OpenGLGetDepthTexture(colorTexture);
//
//    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorTexture, 0);
//    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthTexture, 0);
//
//    // Clear swapchain and depth buffer.
//    glClearColor(DarkSlateGray[0], DarkSlateGray[1], DarkSlateGray[2], DarkSlateGray[3]);
//    glClearDepth(1.0f);
//    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
//
//    // Set shaders and uniform variables.
//    glUseProgram(g_program);
//
//    const auto& pose = layerView.pose;
//    XrMatrix4x4f proj;
//    XrMatrix4x4f_CreateProjectionFov(&proj, GRAPHICS_OPENGL, layerView.fov, 0.05f, 100.0f);
//    XrMatrix4x4f toView;
//    XrVector3f scale{ 1.f, 1.f, 1.f };
//    XrMatrix4x4f_CreateTranslationRotationScale(&toView, &pose.position, &pose.orientation, &scale);
//    XrMatrix4x4f view;
//    XrMatrix4x4f_InvertRigidBody(&view, &toView);
//    XrMatrix4x4f vp;
//    XrMatrix4x4f_Multiply(&vp, &proj, &view);
//
//    // Set cube primitive data.
//    glBindVertexArray(g_vao);
//
//    glBindBuffer(GL_ARRAY_BUFFER, g_cubeVertexBuffer);
//    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_cubeIndexBuffer);
//
//    // Render a cube within each of the spaces we've been asked to render, at the requested sizes.  These show
//    // the centers of each of the spaces we defined.
//    /// @todo Use the name of each space to determine what to draw in it.
//    //int i = 0;
//    for (const Space& space : spaces) {
//        if (g_verbosity >= 10) {
//            std::cout << " Rendering " << space.Name << " space" << std::endl;
//        }
//        // Compute the model-view-projection transform and set it..
//        XrMatrix4x4f model;
//        XrMatrix4x4f_CreateTranslationRotationScale(&model, &space.Pose.position, &space.Pose.orientation, &space.Scale);
//        XrMatrix4x4f mvp;
//        XrMatrix4x4f_Multiply(&mvp, &vp, &model);
//        glUniformMatrix4fv(g_modelViewProjectionUniformLocation, 1, GL_FALSE, reinterpret_cast<const GLfloat*>(&mvp));
//
//        // Draw the cube.
//        glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(ArraySize(Geometry::c_cubeIndices)), GL_UNSIGNED_SHORT, nullptr);
//    }
//
//
//
//    {
//        // draw skybox as last
//        glDepthFunc(GL_LEQUAL);  // change depth function so depth test passes when values are equal to depth buffer's content
//        glUseProgram(g_programSkybox);
//        //view = glm::mat4(glm::mat3(camera.GetViewMatrix())); // remove translation from the view matrix
//       //skyboxShader.setMat4("view", view);
//        glUniformMatrix4fv(glGetUniformLocation(g_programSkybox, "view"), 1, GL_FALSE, reinterpret_cast<const GLfloat*>(&view));
//        //skyboxShader.setMat4("projection", projection);
//        glUniformMatrix4fv(glGetUniformLocation(g_programSkybox, "projection"), 1, GL_FALSE, reinterpret_cast<const GLfloat*>(&proj));
//        // skybox cube
//        glBindVertexArray(skyboxVAO);
//        glActiveTexture(GL_TEXTURE0);
//        glBindTexture(GL_TEXTURE_CUBE_MAP, cubemapTexture);
//        glDrawArrays(GL_TRIANGLES, 0, 36);
//        glBindVertexArray(0);
//        glDepthFunc(GL_LESS); // set depth function back to default
//    }
//
//    // render
//// ------
//    //{
//    //    //glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
//    //    //glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//
//    //    // bind textures on corresponding texture units
//    //    glActiveTexture(GL_TEXTURE0);
//    //    glBindTexture(GL_TEXTURE_2D, textureScene1);
//    //    //glActiveTexture(GL_TEXTURE1);
//    //    //glBindTexture(GL_TEXTURE_2D, texture2);
//
//    //    // activate shader
//    //    glUseProgram(g_programScene1);
//    //    // pass projection matrix to shader (note that in this case it could change every frame)
//    //    glUniformMatrix4fv(glGetUniformLocation(g_programScene1, "projection"), 1, GL_FALSE, reinterpret_cast<const GLfloat*>(&proj));
//
//    //    // camera/view transformation
//    //    glUniformMatrix4fv(glGetUniformLocation(g_programScene1, "view"), 1, GL_FALSE, reinterpret_cast<const GLfloat*>(&view));
//
//    //    // render boxes
//    //    glBindVertexArray(Scene1_vao);
//    //    for (unsigned int i = 0; i < 9; i++)
//    //    {
//    //        // calculate the model matrix for each object and pass it to shader before drawing
//    //        glm::mat4 model = glm::mat4(1.0f); // make sure to initialize matrix to identity matrix first
//    //        model = glm::translate(model, cubePositions[i]);
//    //        float angle = 20.0f * i;
//    //        model = glm::rotate(model, glm::radians(angle), glm::vec3(1.0f, 0.3f, 0.5f));
//    //        glUniformMatrix4fv(glGetUniformLocation(g_programScene1, "model"), 1, GL_FALSE, reinterpret_cast<const GLfloat*>(&model));
//
//    //        glDrawArrays(GL_TRIANGLES, 0, 36);
//    //    }
//
//    //}
//    glUseProgram(g_programScene1);
//    // view/projection transformations
//    //glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
//    //glm::mat4 view = camera.GetViewMatrix();
//    //ourShader.setMat4("projection", projection);
//    glUniformMatrix4fv(glGetUniformLocation(g_programScene1, "projection"), 1, GL_FALSE, reinterpret_cast<const GLfloat*>(&proj));
//    glUniformMatrix4fv(glGetUniformLocation(g_programScene1, "view"), 1, GL_FALSE, reinterpret_cast<const GLfloat*>(&view));
//
//    // render the loaded model
//    glm::mat4 model = glm::mat4(1.0f);
//    model = glm::translate(model, glm::vec3(0.0f, 0.0f, -1.0f)); // translate it down so it's at the center of the scene
//    model = glm::scale(model, glm::vec3(0.5f, 0.5f, 0.5f));	// it's a bit too big for our scene, so scale it down
//    //ourShader.setMat4("model", model);
//    glUniformMatrix4fv(glGetUniformLocation(g_programScene1, "model"), 1, GL_FALSE, reinterpret_cast<const GLfloat*>(&model));
//    //ourModel.Draw(ourShader);
//    ourModelPtr->Draw(g_programScene1);
//
//    GLint viewport[4];
//    glGetIntegerv(GL_VIEWPORT, viewport);
//    int x = viewport[0];
//    int y = viewport[1];
//    int width = viewport[2];
//    int height = viewport[3];
//
//
//    // Read the color framebuffer
//    glBindBuffer(GL_PIXEL_PACK_BUFFER, _PBO[index]);
//    glReadPixels(x, y, width, height, GL_BGR, GL_UNSIGNED_BYTE, 0);
//
//    // Map the color data from the previous frame's PBO
//    glBindBuffer(GL_PIXEL_PACK_BUFFER, _PBO[1 - index]);
//    GLubyte* ptr = (GLubyte*)glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);
//
//    // Read the depth buffer
//    glBindBuffer(GL_PIXEL_PACK_BUFFER, _PBODepth[index]);
//    glReadPixels(x, y, width, height, GL_DEPTH_COMPONENT, GL_FLOAT, 0);
//
//    // Map the depth data from the previous frame's PBO
//    glBindBuffer(GL_PIXEL_PACK_BUFFER, _PBODepth[1 - index]);
//    float* depthPtr = (float*)glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);
//
//
//    //cv::Mat inpainted;
//    if (ptr && depthPtr)
//    {
//        cv::Mat img(height, width, CV_8UC3, ptr);
//        // Process img with OpenCV...
//        cv::flip(img, img, 0); // Flip the image vertically
//        //cv::imshow("OpenGL Frame", img); // Display the image
//        //cv::waitKey(1); // Update the window
//
//        cv::Mat depthMat(height, width, CV_32FC1, depthPtr);
//        cv::Mat depthNormalized;
//        cv::normalize(depthMat, depthNormalized, 0, 255, cv::NORM_MINMAX, CV_8U);
//        depthNormalized = 255 - depthNormalized;
//        cv::flip(depthNormalized, depthNormalized, 0);
//        //cv::imshow("OpenGL Frame", depthNormalized); // Display the image
//        //cv::waitKey(1); // Update the window
//        cv::Mat resized_image;
//        cv::resize(img, resized_image, cv::Size(img.cols, img.rows));
//        cv::Mat resized_depthimage;
//        cv::resize(depthNormalized, resized_depthimage, cv::Size(depthNormalized.cols, depthNormalized.rows));
//
//        cv::Mat shift_image;
//        if (mode) // image warping
//        {
//            shift_image = warp_image_based_on_depth(resized_image, resized_depthimage, 50);
//        }
//
//        else // add manully
//        {
//            img.copyTo(shift_image);
//        }
//
//        //cv::imshow("OpenGL Frame shift", shift_image); // Display the image
//        //cv::waitKey(1); // Update the window
//
//        cv::Mat gray;
//        cv::cvtColor(shift_image, gray, cv::COLOR_BGRA2GRAY);
//
//        // Threshold the grayscale image
//        cv::Mat binary;
//        cv::threshold(gray, binary, 254, 255, cv::THRESH_BINARY_INV); // create mask image
//
//        //cv::imshow("OpenGL Frame", binary); // Display the image
//        //cv::waitKey(1); // Update the window
//
//
//        dr::PixMix pm;
//        dr::det::PixMixParams params;
//        params.alpha = 1.0f;			// 0.0f means no spatial cost considered
//        params.maxItr = 3;				// set to 1 to crank up the speed
//        params.maxRandSearchItr = 3;	// set to 1 to crank up the speed
//        bool debugViz = false;
//        pm.Run(shift_image, binary, inpainted, params, debugViz);
//        //inpainted = img.clone();
//        cv::imshow("Inpainted color image right eye", inpainted);
//        cv::waitKey(1);;
//
//        glBindBuffer(GL_PIXEL_PACK_BUFFER, _PBO[1 - index]);
//        glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
//
//        // Unmap the depth PBO
//        glBindBuffer(GL_PIXEL_PACK_BUFFER, _PBODepth[1 - index]);
//        glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
//    }
//
//    // Unbind the PBO
//    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
//
//
//    // flip index for next frame
//    index = 1 - index;
//
//    glBindVertexArray(0);
//    glUseProgram(0);
//
//    glBindFramebuffer(GL_FRAMEBUFFER, 0);
//
//    // Swap our window every other eye for RenderDoc
//    static int everyOther = 0;
//    if ((everyOther++ & 1) != 0) {
//        ksGpuWindow_SwapBuffers(&g_window);
//    }
//}

static void OpenGLTearDown()
{
    if (g_swapchainFramebuffer != 0) {
        glDeleteFramebuffers(1, &g_swapchainFramebuffer);
    }
    if (g_program != 0) {
        glDeleteProgram(g_program);
    }
    if (g_vao != 0) {
        glDeleteVertexArrays(1, &g_vao);
    }
    if (g_cubeVertexBuffer != 0) {
        glDeleteBuffers(1, &g_cubeVertexBuffer);
    }
    if (g_cubeIndexBuffer != 0) {
        glDeleteBuffers(1, &g_cubeIndexBuffer);
    }

    for (auto& colorToDepth : g_colorToDepthMap) {
        if (colorToDepth.second != 0) {
            glDeleteTextures(1, &colorToDepth.second);
        }
    }
} 


//============================================================================================
// OpenXR state and functions.


static void OpenXRCreateInstance()
{
#ifdef XR_USE_PLATFORM_WIN32
    CHECK_HRCMD(CoInitializeEx(nullptr, COINIT_MULTITHREADED));
#endif

    CHECK(g_instance == XR_NULL_HANDLE);

    // Create union of extensions required by OpenGL.
    std::vector<const char*> extensions = {XR_KHR_OPENGL_ENABLE_EXTENSION_NAME};

    // We'll get a list of extensions that OpenXR provides using this 
    // enumerate pattern. OpenXR often uses a two-call enumeration pattern 
    // where the first call will tell you how much memory to allocate, and
    // the second call will provide you with the actual data!
    uint32_t ext_count = 0;
    xrEnumerateInstanceExtensionProperties(nullptr, 0, &ext_count, nullptr);
    std::vector<XrExtensionProperties> xr_exts(ext_count, { XR_TYPE_EXTENSION_PROPERTIES });
    xrEnumerateInstanceExtensionProperties(nullptr, ext_count, &ext_count, xr_exts.data());

    printf("OpenXR extensions available:\n");
    for (size_t i = 0; i < xr_exts.size(); i++) {
        printf("- %s\n", xr_exts[i].extensionName);

        // Check if we're asking for this extensions, and add it to our use 
        // list!
        //for (int32_t ask = 0; ask < _countof(ask_extensions); ask++) {
        //    if (strcmp(ask_extensions[ask], xr_exts[i].extensionName) == 0) {
        //        extensions.push_back(ask_extensions[ask]);
        //        break;
        //    }
        //}
    }



    XrInstanceCreateInfo createInfo{XR_TYPE_INSTANCE_CREATE_INFO};
    createInfo.next = nullptr;  // Needs to be set on Android.
    createInfo.enabledExtensionCount = (uint32_t)extensions.size();
    createInfo.enabledExtensionNames = extensions.data();

    /// @todo Change the application name here.
    strcpy(createInfo.applicationInfo.applicationName, "OpenXR-OpenGL-Example");
    createInfo.applicationInfo.apiVersion = XR_CURRENT_API_VERSION;

    CHECK_XRCMD(xrCreateInstance(&createInfo, &g_instance));
}

static XrReferenceSpaceCreateInfo GetXrReferenceSpaceCreateInfo(const std::string& referenceSpaceTypeStr) {
    XrReferenceSpaceCreateInfo referenceSpaceCreateInfo{XR_TYPE_REFERENCE_SPACE_CREATE_INFO};
    referenceSpaceCreateInfo.poseInReferenceSpace = Math::Pose::Identity();
    if (EqualsIgnoreCase(referenceSpaceTypeStr, "View")) {
        referenceSpaceCreateInfo.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_VIEW;
    } else if (EqualsIgnoreCase(referenceSpaceTypeStr, "ViewFront")) {
        // Render head-locked 2m in front of device.
        referenceSpaceCreateInfo.poseInReferenceSpace = Math::Pose::Translation({0.f, 0.f, -2.f}),
        referenceSpaceCreateInfo.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_VIEW;
    } else if (EqualsIgnoreCase(referenceSpaceTypeStr, "Local")) {
        referenceSpaceCreateInfo.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_LOCAL;
    } else if (EqualsIgnoreCase(referenceSpaceTypeStr, "Stage")) {
        referenceSpaceCreateInfo.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_STAGE;
    } else if (EqualsIgnoreCase(referenceSpaceTypeStr, "StageLeft")) {
        referenceSpaceCreateInfo.poseInReferenceSpace = Math::Pose::RotateCCWAboutYAxis(0.f, {-2.f, 0.f, -2.f});
        referenceSpaceCreateInfo.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_STAGE;
    } else if (EqualsIgnoreCase(referenceSpaceTypeStr, "StageRight")) {
        referenceSpaceCreateInfo.poseInReferenceSpace = Math::Pose::RotateCCWAboutYAxis(0.f, {2.f, 0.f, -2.f});
        referenceSpaceCreateInfo.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_STAGE;
    } else if (EqualsIgnoreCase(referenceSpaceTypeStr, "StageLeftRotated")) {
        referenceSpaceCreateInfo.poseInReferenceSpace = Math::Pose::RotateCCWAboutYAxis(3.14f / 3.f, {-2.f, 0.5f, -2.f});
        referenceSpaceCreateInfo.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_STAGE;
    } else if (EqualsIgnoreCase(referenceSpaceTypeStr, "StageRightRotated")) {
        referenceSpaceCreateInfo.poseInReferenceSpace = Math::Pose::RotateCCWAboutYAxis(-3.14f / 3.f, {2.f, 0.5f, -2.f});
        referenceSpaceCreateInfo.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_STAGE;
    } else {
        throw std::invalid_argument(Fmt("Unknown reference space type '%s'", referenceSpaceTypeStr.c_str()));
    }
    return referenceSpaceCreateInfo;
}

/// @todo Change these to match the desired behavior.
struct Options {
    std::string GraphicsPlugin;
    XrFormFactor FormFactor{XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY};
    XrViewConfigurationType ViewConfiguration{XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO};
    XrEnvironmentBlendMode EnvironmentBlendMode{XR_ENVIRONMENT_BLEND_MODE_OPAQUE};
    std::string AppSpace{"Local"};
} g_options;

static void OpenXRInitializeSystem()
{
        CHECK(g_instance != XR_NULL_HANDLE);
        CHECK(g_systemId == XR_NULL_SYSTEM_ID);

        g_formFactor = g_options.FormFactor;
        g_viewConfigType = g_options.ViewConfiguration;
        g_environmentBlendMode = g_options.EnvironmentBlendMode;

        XrSystemGetInfo systemInfo{XR_TYPE_SYSTEM_GET_INFO};
        systemInfo.formFactor = g_formFactor;
        CHECK_XRCMD(xrGetSystem(g_instance, &systemInfo, &g_systemId));

        if (g_verbosity >= 2) std::cout << "Using system " << g_systemId
            << " for form factor " <<  to_string(g_formFactor) << std::endl;
        CHECK(g_instance != XR_NULL_HANDLE);
        CHECK(g_systemId != XR_NULL_SYSTEM_ID);

        /// @todo Print information about the system here in verbose mode.

        // The graphics API can initialize the graphics device now that the systemId and instance
        // handle are available.
        OpenGLInitializeDevice(g_instance, g_systemId);
}

/// @todo Change the behaviors by modifying the action bindings.

void OpenXRInitializeActions() {
    // Create an action set.
    {
        XrActionSetCreateInfo actionSetInfo{XR_TYPE_ACTION_SET_CREATE_INFO};
        strcpy_s(actionSetInfo.actionSetName, "gameplay");
        strcpy_s(actionSetInfo.localizedActionSetName, "Gameplay");
        actionSetInfo.priority = 0;
        CHECK_XRCMD(xrCreateActionSet(g_instance, &actionSetInfo, &g_input.actionSet));
    }

    // Get the XrPath for the left and right hands - we will use them as subaction paths.
    CHECK_XRCMD(xrStringToPath(g_instance, "/user/hand/left", &g_input.handSubactionPath[Side::LEFT]));
    CHECK_XRCMD(xrStringToPath(g_instance, "/user/hand/right", &g_input.handSubactionPath[Side::RIGHT]));

    // Create actions.
    {
        // Create an input action for grabbing objects with the left and right hands.
        XrActionCreateInfo actionInfo{XR_TYPE_ACTION_CREATE_INFO};
        actionInfo.actionType = XR_ACTION_TYPE_FLOAT_INPUT;
        strcpy_s(actionInfo.actionName, "grab_object");
        strcpy_s(actionInfo.localizedActionName, "Grab Object");
        actionInfo.countSubactionPaths = uint32_t(g_input.handSubactionPath.size());
        actionInfo.subactionPaths = g_input.handSubactionPath.data();
        CHECK_XRCMD(xrCreateAction(g_input.actionSet, &actionInfo, &g_input.grabAction));

        // Create an input action getting the left and right hand poses.
        actionInfo.actionType = XR_ACTION_TYPE_POSE_INPUT;
        strcpy_s(actionInfo.actionName, "hand_pose");
        strcpy_s(actionInfo.localizedActionName, "Hand Pose");
        actionInfo.countSubactionPaths = uint32_t(g_input.handSubactionPath.size());
        actionInfo.subactionPaths = g_input.handSubactionPath.data();
        CHECK_XRCMD(xrCreateAction(g_input.actionSet, &actionInfo, &g_input.poseAction));

        // Create output actions for vibrating the left and right controller.
        actionInfo.actionType = XR_ACTION_TYPE_VIBRATION_OUTPUT;
        strcpy_s(actionInfo.actionName, "vibrate_hand");
        strcpy_s(actionInfo.localizedActionName, "Vibrate Hand");
        actionInfo.countSubactionPaths = uint32_t(g_input.handSubactionPath.size());
        actionInfo.subactionPaths = g_input.handSubactionPath.data();
        CHECK_XRCMD(xrCreateAction(g_input.actionSet, &actionInfo, &g_input.vibrateAction));

        // Create input actions for quitting the session using the left and right controller.
        // Since it doesn't matter which hand did this, we do not specify subaction paths for it.
        // We will just suggest bindings for both hands, where possible.
        actionInfo.actionType = XR_ACTION_TYPE_BOOLEAN_INPUT;
        strcpy_s(actionInfo.actionName, "quit_session");
        strcpy_s(actionInfo.localizedActionName, "Quit Session");
        actionInfo.countSubactionPaths = 0;
        actionInfo.subactionPaths = nullptr;
        CHECK_XRCMD(xrCreateAction(g_input.actionSet, &actionInfo, &g_input.quitAction));
    }

    std::array<XrPath, Side::COUNT> selectPath;
    std::array<XrPath, Side::COUNT> squeezeValuePath;
    std::array<XrPath, Side::COUNT> squeezeForcePath;
    std::array<XrPath, Side::COUNT> squeezeClickPath;
    std::array<XrPath, Side::COUNT> posePath;
    std::array<XrPath, Side::COUNT> hapticPath;
    std::array<XrPath, Side::COUNT> menuClickPath;
    std::array<XrPath, Side::COUNT> bClickPath;
    std::array<XrPath, Side::COUNT> triggerValuePath;
    CHECK_XRCMD(xrStringToPath(g_instance, "/user/hand/left/input/select/click", &selectPath[Side::LEFT]));
    CHECK_XRCMD(xrStringToPath(g_instance, "/user/hand/right/input/select/click", &selectPath[Side::RIGHT]));
    CHECK_XRCMD(xrStringToPath(g_instance, "/user/hand/left/input/squeeze/value", &squeezeValuePath[Side::LEFT]));
    CHECK_XRCMD(xrStringToPath(g_instance, "/user/hand/right/input/squeeze/value", &squeezeValuePath[Side::RIGHT]));
    CHECK_XRCMD(xrStringToPath(g_instance, "/user/hand/left/input/squeeze/force", &squeezeForcePath[Side::LEFT]));
    CHECK_XRCMD(xrStringToPath(g_instance, "/user/hand/right/input/squeeze/force", &squeezeForcePath[Side::RIGHT]));
    CHECK_XRCMD(xrStringToPath(g_instance, "/user/hand/left/input/squeeze/click", &squeezeClickPath[Side::LEFT]));
    CHECK_XRCMD(xrStringToPath(g_instance, "/user/hand/right/input/squeeze/click", &squeezeClickPath[Side::RIGHT]));
    CHECK_XRCMD(xrStringToPath(g_instance, "/user/hand/left/input/grip/pose", &posePath[Side::LEFT]));
    CHECK_XRCMD(xrStringToPath(g_instance, "/user/hand/right/input/grip/pose", &posePath[Side::RIGHT]));
    CHECK_XRCMD(xrStringToPath(g_instance, "/user/hand/left/output/haptic", &hapticPath[Side::LEFT]));
    CHECK_XRCMD(xrStringToPath(g_instance, "/user/hand/right/output/haptic", &hapticPath[Side::RIGHT]));
    CHECK_XRCMD(xrStringToPath(g_instance, "/user/hand/left/input/menu/click", &menuClickPath[Side::LEFT]));
    CHECK_XRCMD(xrStringToPath(g_instance, "/user/hand/right/input/menu/click", &menuClickPath[Side::RIGHT]));
    CHECK_XRCMD(xrStringToPath(g_instance, "/user/hand/left/input/b/click", &bClickPath[Side::LEFT]));
    CHECK_XRCMD(xrStringToPath(g_instance, "/user/hand/right/input/b/click", &bClickPath[Side::RIGHT]));
    CHECK_XRCMD(xrStringToPath(g_instance, "/user/hand/left/input/trigger/value", &triggerValuePath[Side::LEFT]));
    CHECK_XRCMD(xrStringToPath(g_instance, "/user/hand/right/input/trigger/value", &triggerValuePath[Side::RIGHT]));
    // Suggest bindings for KHR Simple.
    {
        XrPath khrSimpleInteractionProfilePath;
        CHECK_XRCMD(
            xrStringToPath(g_instance, "/interaction_profiles/khr/simple_controller", &khrSimpleInteractionProfilePath));
        std::vector<XrActionSuggestedBinding> bindings{{// Fall back to a click input for the grab action.
                                                        {g_input.grabAction, selectPath[Side::LEFT]},
                                                        {g_input.grabAction, selectPath[Side::RIGHT]},
                                                        {g_input.poseAction, posePath[Side::LEFT]},
                                                        {g_input.poseAction, posePath[Side::RIGHT]},
                                                        {g_input.quitAction, menuClickPath[Side::LEFT]},
                                                        {g_input.quitAction, menuClickPath[Side::RIGHT]},
                                                        {g_input.vibrateAction, hapticPath[Side::LEFT]},
                                                        {g_input.vibrateAction, hapticPath[Side::RIGHT]}}};
        XrInteractionProfileSuggestedBinding suggestedBindings{XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING};
        suggestedBindings.interactionProfile = khrSimpleInteractionProfilePath;
        suggestedBindings.suggestedBindings = bindings.data();
        suggestedBindings.countSuggestedBindings = (uint32_t)bindings.size();
        CHECK_XRCMD(xrSuggestInteractionProfileBindings(g_instance, &suggestedBindings));
    }
    // Suggest bindings for the Oculus Touch.
    {
        XrPath oculusTouchInteractionProfilePath;
        CHECK_XRCMD(
            xrStringToPath(g_instance, "/interaction_profiles/oculus/touch_controller", &oculusTouchInteractionProfilePath));
        std::vector<XrActionSuggestedBinding> bindings{{{g_input.grabAction, squeezeValuePath[Side::LEFT]},
                                                        {g_input.grabAction, squeezeValuePath[Side::RIGHT]},
                                                        {g_input.poseAction, posePath[Side::LEFT]},
                                                        {g_input.poseAction, posePath[Side::RIGHT]},
                                                        {g_input.quitAction, menuClickPath[Side::LEFT]},
                                                        {g_input.vibrateAction, hapticPath[Side::LEFT]},
                                                        {g_input.vibrateAction, hapticPath[Side::RIGHT]}}};
        XrInteractionProfileSuggestedBinding suggestedBindings{XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING};
        suggestedBindings.interactionProfile = oculusTouchInteractionProfilePath;
        suggestedBindings.suggestedBindings = bindings.data();
        suggestedBindings.countSuggestedBindings = (uint32_t)bindings.size();
        CHECK_XRCMD(xrSuggestInteractionProfileBindings(g_instance, &suggestedBindings));
    }
    // Suggest bindings for the Vive Controller.
    {
        XrPath viveControllerInteractionProfilePath;
        CHECK_XRCMD(
            xrStringToPath(g_instance, "/interaction_profiles/htc/vive_controller", &viveControllerInteractionProfilePath));
        std::vector<XrActionSuggestedBinding> bindings{{{g_input.grabAction, triggerValuePath[Side::LEFT]},
                                                        {g_input.grabAction, triggerValuePath[Side::RIGHT]},
                                                        {g_input.poseAction, posePath[Side::LEFT]},
                                                        {g_input.poseAction, posePath[Side::RIGHT]},
                                                        {g_input.quitAction, menuClickPath[Side::LEFT]},
                                                        {g_input.quitAction, menuClickPath[Side::RIGHT]},
                                                        {g_input.vibrateAction, hapticPath[Side::LEFT]},
                                                        {g_input.vibrateAction, hapticPath[Side::RIGHT]}}};
        XrInteractionProfileSuggestedBinding suggestedBindings{XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING};
        suggestedBindings.interactionProfile = viveControllerInteractionProfilePath;
        suggestedBindings.suggestedBindings = bindings.data();
        suggestedBindings.countSuggestedBindings = (uint32_t)bindings.size();
        CHECK_XRCMD(xrSuggestInteractionProfileBindings(g_instance, &suggestedBindings));
    }

    // Suggest bindings for the Valve Index Controller.
    {
        XrPath indexControllerInteractionProfilePath;
        CHECK_XRCMD(
            xrStringToPath(g_instance, "/interaction_profiles/valve/index_controller", &indexControllerInteractionProfilePath));
        std::vector<XrActionSuggestedBinding> bindings{{{g_input.grabAction, squeezeForcePath[Side::LEFT]},
                                                        {g_input.grabAction, squeezeForcePath[Side::RIGHT]},
                                                        {g_input.poseAction, posePath[Side::LEFT]},
                                                        {g_input.poseAction, posePath[Side::RIGHT]},
                                                        {g_input.quitAction, bClickPath[Side::LEFT]},
                                                        {g_input.quitAction, bClickPath[Side::RIGHT]},
                                                        {g_input.vibrateAction, hapticPath[Side::LEFT]},
                                                        {g_input.vibrateAction, hapticPath[Side::RIGHT]}}};
        XrInteractionProfileSuggestedBinding suggestedBindings{XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING};
        suggestedBindings.interactionProfile = indexControllerInteractionProfilePath;
        suggestedBindings.suggestedBindings = bindings.data();
        suggestedBindings.countSuggestedBindings = (uint32_t)bindings.size();
        CHECK_XRCMD(xrSuggestInteractionProfileBindings(g_instance, &suggestedBindings));
    }

    // Suggest bindings for the Microsoft Mixed Reality Motion Controller.
    {
        XrPath microsoftMixedRealityInteractionProfilePath;
        CHECK_XRCMD(xrStringToPath(g_instance, "/interaction_profiles/microsoft/motion_controller",
                                   &microsoftMixedRealityInteractionProfilePath));
        std::vector<XrActionSuggestedBinding> bindings{{{g_input.grabAction, squeezeClickPath[Side::LEFT]},
                                                        {g_input.grabAction, squeezeClickPath[Side::RIGHT]},
                                                        {g_input.poseAction, posePath[Side::LEFT]},
                                                        {g_input.poseAction, posePath[Side::RIGHT]},
                                                        {g_input.quitAction, menuClickPath[Side::LEFT]},
                                                        {g_input.quitAction, menuClickPath[Side::RIGHT]},
                                                        {g_input.vibrateAction, hapticPath[Side::LEFT]},
                                                        {g_input.vibrateAction, hapticPath[Side::RIGHT]}}};
        XrInteractionProfileSuggestedBinding suggestedBindings{XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING};
        suggestedBindings.interactionProfile = microsoftMixedRealityInteractionProfilePath;
        suggestedBindings.suggestedBindings = bindings.data();
        suggestedBindings.countSuggestedBindings = (uint32_t)bindings.size();
        CHECK_XRCMD(xrSuggestInteractionProfileBindings(g_instance, &suggestedBindings));
    }
    XrActionSpaceCreateInfo actionSpaceInfo{XR_TYPE_ACTION_SPACE_CREATE_INFO};
    actionSpaceInfo.action = g_input.poseAction;
    actionSpaceInfo.poseInActionSpace.orientation.w = 1.f;
    actionSpaceInfo.subactionPath = g_input.handSubactionPath[Side::LEFT];
    CHECK_XRCMD(xrCreateActionSpace(g_session, &actionSpaceInfo, &g_input.handSpace[Side::LEFT]));
    actionSpaceInfo.subactionPath = g_input.handSubactionPath[Side::RIGHT];
    CHECK_XRCMD(xrCreateActionSpace(g_session, &actionSpaceInfo, &g_input.handSpace[Side::RIGHT]));

    XrSessionActionSetsAttachInfo attachInfo{XR_TYPE_SESSION_ACTION_SETS_ATTACH_INFO};
    attachInfo.countActionSets = 1;
    attachInfo.actionSets = &g_input.actionSet;
    CHECK_XRCMD(xrAttachSessionActionSets(g_session, &attachInfo));
}


void OpenXRCreateVisualizedSpaces() {
    CHECK(g_session != XR_NULL_HANDLE);

    /// @todo Change this to modify the spaces that have things drawn in them.  They can all be removed
    /// if you draw things in world space.  Removing these will not remove the cubes drawn for the hands.
    std::string visualizedSpaces[] = {"ViewFront", "Local", "Stage", "StageLeft", "StageRight", "StageLeftRotated",
                                      "StageRightRotated"};

    for (const auto& visualizedSpace : visualizedSpaces) {
        XrReferenceSpaceCreateInfo referenceSpaceCreateInfo = GetXrReferenceSpaceCreateInfo(visualizedSpace);
        XrSpace space;
        XrResult res = xrCreateReferenceSpace(g_session, &referenceSpaceCreateInfo, &space);
        if (XR_SUCCEEDED(res)) {
            g_visualizedSpaces.push_back(space);
            g_spaceNames[space] = visualizedSpace;
        } else {
            if (g_verbosity >= 0) {
                std::cerr << "Failed to create reference space " << visualizedSpace << " with error " << res
                    << std::endl;
            }
        }
    }
}


void OpenXRInitializeSession()
{
    CHECK(g_instance != XR_NULL_HANDLE);
    CHECK(g_session == XR_NULL_HANDLE);

    {
        if (g_verbosity >= 2) std::cout << Fmt("Creating session...") << std::endl;

        XrSessionCreateInfo createInfo{XR_TYPE_SESSION_CREATE_INFO};
        createInfo.next = reinterpret_cast<const XrBaseInStructure*>(&g_graphicsBinding);
        createInfo.systemId = g_systemId;
        CHECK_XRCMD(xrCreateSession(g_instance, &createInfo, &g_session));
    }

    /// @todo Print the reference spaces.

    OpenXRInitializeActions();
    OpenXRCreateVisualizedSpaces();

    {
        XrReferenceSpaceCreateInfo referenceSpaceCreateInfo = GetXrReferenceSpaceCreateInfo(g_options.AppSpace);
        CHECK_XRCMD(xrCreateReferenceSpace(g_session, &referenceSpaceCreateInfo, &g_appSpace));
    }
}

static void OpenXRCreateSwapchains()
{
    CHECK(g_session != XR_NULL_HANDLE);
    CHECK(g_swapchains.empty());
    CHECK(g_configViews.empty());

    // Read graphics properties for preferred swapchain length and logging.
    XrSystemProperties systemProperties{XR_TYPE_SYSTEM_PROPERTIES};
    CHECK_XRCMD(xrGetSystemProperties(g_instance, g_systemId, &systemProperties));

    // Log system properties.
    if (g_verbosity >= 1) {
        std::cout <<
                   Fmt("System Properties: Name=%s VendorId=%d", systemProperties.systemName, systemProperties.vendorId)
            << std::endl;
        std::cout << Fmt("System Graphics Properties: MaxWidth=%d MaxHeight=%d MaxLayers=%d",
                                         systemProperties.graphicsProperties.maxSwapchainImageWidth,
                                         systemProperties.graphicsProperties.maxSwapchainImageHeight,
                                         systemProperties.graphicsProperties.maxLayerCount)
            << std::endl;
        std::cout << Fmt("System Tracking Properties: OrientationTracking=%s PositionTracking=%s",
                                         systemProperties.trackingProperties.orientationTracking == XR_TRUE ? "True" : "False",
                                         systemProperties.trackingProperties.positionTracking == XR_TRUE ? "True" : "False")
            << std::endl;
    }

    // Note: No other view configurations exist at the time this code was written. If this
    // condition is not met, the project will need to be audited to see how support should be
    // added.
    CHECK_MSG(g_viewConfigType == XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, "Unsupported view configuration type");

    // Query and cache view configuration views.
    uint32_t viewCount;
    CHECK_XRCMD(xrEnumerateViewConfigurationViews(g_instance, g_systemId, g_viewConfigType, 0, &viewCount, nullptr));
    g_configViews.resize(viewCount, {XR_TYPE_VIEW_CONFIGURATION_VIEW});
    CHECK_XRCMD(xrEnumerateViewConfigurationViews(g_instance, g_systemId, g_viewConfigType, viewCount, &viewCount,
                                                  g_configViews.data()));

    // Create and cache view buffer for xrLocateViews later.
    g_views.resize(viewCount, {XR_TYPE_VIEW});

    // Create the swapchain and get the images.
    if (viewCount > 0) {
        // Select a swapchain format.
        uint32_t swapchainFormatCount;
        CHECK_XRCMD(xrEnumerateSwapchainFormats(g_session, 0, &swapchainFormatCount, nullptr));
        std::vector<int64_t> swapchainFormats(swapchainFormatCount);
        CHECK_XRCMD(xrEnumerateSwapchainFormats(g_session, (uint32_t)swapchainFormats.size(), &swapchainFormatCount,
                                                swapchainFormats.data()));
        CHECK(swapchainFormatCount == swapchainFormats.size());
        g_colorSwapchainFormat = OpenGLSelectColorSwapchainFormat(swapchainFormats);

        // Print swapchain formats and the selected one.
        {
            std::string swapchainFormatsString;
            for (int64_t format : swapchainFormats) {
                const bool selected = format == g_colorSwapchainFormat;
                swapchainFormatsString += " ";
                if (selected) {
                    swapchainFormatsString += "[";
                }
                swapchainFormatsString += std::to_string(format);
                if (selected) {
                    swapchainFormatsString += "]";
                }
            }
            if (g_verbosity >= 1) std::cout << Fmt("Swapchain Formats: %s", swapchainFormatsString.c_str()) << std::endl;
        }

        // Create a swapchain for each view.
        for (uint32_t i = 0; i < viewCount; i++) {
            const XrViewConfigurationView& vp = g_configViews[i];
            if (g_verbosity >= 1) {
                std::cout << 
                   Fmt("Creating swapchain for view %d with dimensions Width=%d Height=%d SampleCount=%d", i,
                       vp.recommendedImageRectWidth, vp.recommendedImageRectHeight, vp.recommendedSwapchainSampleCount)
                    << std::endl;
            }

            // Create the swapchain.
            XrSwapchainCreateInfo swapchainCreateInfo{XR_TYPE_SWAPCHAIN_CREATE_INFO};
            swapchainCreateInfo.arraySize = 1;
            swapchainCreateInfo.format = g_colorSwapchainFormat;
            swapchainCreateInfo.width = vp.recommendedImageRectWidth/2;
            swapchainCreateInfo.height = vp.recommendedImageRectHeight/2;
            std::cout <<"swapchain width"<< swapchainCreateInfo.width << std::endl;
            std::cout << "swapchain height" << swapchainCreateInfo.height << std::endl;
            swapchainCreateInfo.mipCount = 1;
            swapchainCreateInfo.faceCount = 1;
            swapchainCreateInfo.sampleCount = vp.recommendedSwapchainSampleCount;
            swapchainCreateInfo.usageFlags = XR_SWAPCHAIN_USAGE_SAMPLED_BIT | XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT;
            Swapchain swapchain;
            swapchain.width = swapchainCreateInfo.width;
            swapchain.height = swapchainCreateInfo.height;
            CHECK_XRCMD(xrCreateSwapchain(g_session, &swapchainCreateInfo, &swapchain.handle));

            g_swapchains.push_back(swapchain);

            uint32_t imageCount;
            CHECK_XRCMD(xrEnumerateSwapchainImages(swapchain.handle, 0, &imageCount, nullptr));
            // XXX This should really just return XrSwapchainImageBaseHeader*
            std::vector<XrSwapchainImageBaseHeader*> swapchainImages =
                OpenGLAllocateSwapchainImageStructs(imageCount, swapchainCreateInfo);
            CHECK_XRCMD(xrEnumerateSwapchainImages(swapchain.handle, imageCount, &imageCount, swapchainImages[0]));

            g_swapchainImages.insert(std::make_pair(swapchain.handle, std::move(swapchainImages)));
        }
    }
}

// Return event if one is available, otherwise return null.
static XrEventDataBaseHeader* OpenXRTryReadNextEvent() {
    // It is sufficient to clear the just the XrEventDataBuffer header to
    // XR_TYPE_EVENT_DATA_BUFFER
    XrEventDataBaseHeader* baseHeader = reinterpret_cast<XrEventDataBaseHeader*>(&g_eventDataBuffer);
    *baseHeader = {XR_TYPE_EVENT_DATA_BUFFER};
    const XrResult xr = xrPollEvent(g_instance, &g_eventDataBuffer);
    if (xr == XR_SUCCESS) {
        if (baseHeader->type == XR_TYPE_EVENT_DATA_EVENTS_LOST) {
            const XrEventDataEventsLost* const eventsLost = reinterpret_cast<const XrEventDataEventsLost*>(baseHeader);
            if (g_verbosity > 0) std::cerr << Fmt("%d events lost", eventsLost) << std::endl;
        }

        return baseHeader;
    }
    if (xr == XR_EVENT_UNAVAILABLE) {
        return nullptr;
    }
    THROW_XR(xr, "xrPollEvent");
}

static void OpenXRHandleSessionStateChangedEvent(const XrEventDataSessionStateChanged& stateChangedEvent, bool* exitRenderLoop,
                                    bool* requestRestart) {
    const XrSessionState oldState = g_sessionState;
    g_sessionState = stateChangedEvent.state;

    if (g_verbosity >= 1) {
        std::cout << Fmt("XrEventDataSessionStateChanged: state %s->%s session=%lld time=%lld", to_string(oldState),
                                     to_string(g_sessionState), stateChangedEvent.session, stateChangedEvent.time)
            << std::endl;
    }

    if ((stateChangedEvent.session != XR_NULL_HANDLE) && (stateChangedEvent.session != g_session)) {
        std::cerr << "XrEventDataSessionStateChanged for unknown session" << std::endl;
        return;
    }

    switch (g_sessionState) {
        case XR_SESSION_STATE_READY: {
            CHECK(g_session != XR_NULL_HANDLE);
            XrSessionBeginInfo sessionBeginInfo{XR_TYPE_SESSION_BEGIN_INFO};
            sessionBeginInfo.primaryViewConfigurationType = g_viewConfigType;
            CHECK_XRCMD(xrBeginSession(g_session, &sessionBeginInfo));
            g_sessionRunning = true;
            break;
        }
        case XR_SESSION_STATE_STOPPING: {
            CHECK(g_session != XR_NULL_HANDLE);
            g_sessionRunning = false;
            CHECK_XRCMD(xrEndSession(g_session))
            break;
        }
        case XR_SESSION_STATE_EXITING: {
            *exitRenderLoop = true;
            // Do not attempt to restart because user closed this session.
            *requestRestart = false;
            break;
        }
        case XR_SESSION_STATE_LOSS_PENDING: {
            *exitRenderLoop = true;
            // Poll for a new instance.
            *requestRestart = true;
            break;
        }
        default:
            break;
    }
}


static void OpenXRPollEvents(bool* exitRenderLoop, bool* requestRestart) {
    *exitRenderLoop = *requestRestart = false;

    // Process all pending messages.
    while (const XrEventDataBaseHeader* event = OpenXRTryReadNextEvent()) {
        switch (event->type) {
            case XR_TYPE_EVENT_DATA_INSTANCE_LOSS_PENDING: {
                const auto& instanceLossPending = *reinterpret_cast<const XrEventDataInstanceLossPending*>(event);
                if (g_verbosity > 0) std::cerr << Fmt("XrEventDataInstanceLossPending by %lld", instanceLossPending.lossTime) << std::endl;
                *exitRenderLoop = true;
                *requestRestart = true;
                return;
            }
            case XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED: {
                auto sessionStateChangedEvent = *reinterpret_cast<const XrEventDataSessionStateChanged*>(event);
                OpenXRHandleSessionStateChangedEvent(sessionStateChangedEvent, exitRenderLoop, requestRestart);
                break;
            }
            case XR_TYPE_EVENT_DATA_INTERACTION_PROFILE_CHANGED:
                break;
            case XR_TYPE_EVENT_DATA_REFERENCE_SPACE_CHANGE_PENDING:
            default: {
                if (g_verbosity >= 2) std::cout << Fmt("Ignoring event type %d", event->type) << std::endl;
                break;
            }
        }
    }
}

static void OpenXRPollActions() {
    g_input.handActive = {XR_FALSE, XR_FALSE};

    // Sync actions
    const XrActiveActionSet activeActionSet{g_input.actionSet, XR_NULL_PATH};
    XrActionsSyncInfo syncInfo{XR_TYPE_ACTIONS_SYNC_INFO};
    syncInfo.countActiveActionSets = 1;
    syncInfo.activeActionSets = &activeActionSet;
    CHECK_XRCMD(xrSyncActions(g_session, &syncInfo));

    // Get pose and grab action state and start haptic vibrate when hand is 90% squeezed.
    for (auto hand : {Side::LEFT, Side::RIGHT}) {
        XrActionStateGetInfo getInfo{XR_TYPE_ACTION_STATE_GET_INFO};
        getInfo.action = g_input.grabAction;
        getInfo.subactionPath = g_input.handSubactionPath[hand];

        XrActionStateFloat grabValue{XR_TYPE_ACTION_STATE_FLOAT};
        CHECK_XRCMD(xrGetActionStateFloat(g_session, &getInfo, &grabValue));
        if (grabValue.isActive == XR_TRUE) {
            // Scale the rendered hand by 1.0f (open) to 0.5f (fully squeezed).
            g_input.handScale[hand] = 1.0f - 0.5f * grabValue.currentState;
            if (grabValue.currentState > 0.9f) {
                XrHapticVibration vibration{XR_TYPE_HAPTIC_VIBRATION};
                vibration.amplitude = 0.5;
                vibration.duration = XR_MIN_HAPTIC_DURATION;
                vibration.frequency = XR_FREQUENCY_UNSPECIFIED;

                XrHapticActionInfo hapticActionInfo{XR_TYPE_HAPTIC_ACTION_INFO};
                hapticActionInfo.action = g_input.vibrateAction;
                hapticActionInfo.subactionPath = g_input.handSubactionPath[hand];
                CHECK_XRCMD(xrApplyHapticFeedback(g_session, &hapticActionInfo, (XrHapticBaseHeader*)&vibration));
            }
        }

        getInfo.action = g_input.poseAction;
        XrActionStatePose poseState{XR_TYPE_ACTION_STATE_POSE};
        CHECK_XRCMD(xrGetActionStatePose(g_session, &getInfo, &poseState));
        g_input.handActive[hand] = poseState.isActive;
    }

    // There were no subaction paths specified for the quit action, because we don't care which hand did it.
    XrActionStateGetInfo getInfo{XR_TYPE_ACTION_STATE_GET_INFO, nullptr, g_input.quitAction, XR_NULL_PATH};
    XrActionStateBoolean quitValue{XR_TYPE_ACTION_STATE_BOOLEAN};
    CHECK_XRCMD(xrGetActionStateBoolean(g_session, &getInfo, &quitValue));
    if ((quitValue.isActive == XR_TRUE) && (quitValue.changedSinceLastSync == XR_TRUE) && (quitValue.currentState == XR_TRUE)) {
        CHECK_XRCMD(xrRequestExitSession(g_session));
    }
}

static bool OpenXRRenderLayer(XrTime predictedDisplayTime, std::vector<XrCompositionLayerProjectionView>& projectionLayerViews,
                 XrCompositionLayerProjection& layer)
{
    XrResult res;

    XrViewState viewState{XR_TYPE_VIEW_STATE};
    uint32_t viewCapacityInput = (uint32_t)g_views.size();
    uint32_t viewCountOutput;

    XrViewLocateInfo viewLocateInfo{XR_TYPE_VIEW_LOCATE_INFO};
    viewLocateInfo.viewConfigurationType = g_viewConfigType;
    viewLocateInfo.displayTime = predictedDisplayTime;
    viewLocateInfo.space = g_appSpace;

    res = xrLocateViews(g_session, &viewLocateInfo, &viewState, viewCapacityInput, &viewCountOutput, g_views.data());
    CHECK_XRRESULT(res, "xrLocateViews");
    if ((viewState.viewStateFlags & XR_VIEW_STATE_POSITION_VALID_BIT) == 0 ||
        (viewState.viewStateFlags & XR_VIEW_STATE_ORIENTATION_VALID_BIT) == 0) {
        return false;  // There is no valid tracking poses for the views.
    }

    CHECK(viewCountOutput == viewCapacityInput);
    CHECK(viewCountOutput == g_configViews.size());
    CHECK(viewCountOutput == g_swapchains.size());

    projectionLayerViews.resize(viewCountOutput);

    // For each locatable space that we want to visualize, render a 25cm cube.
    std::vector<Space> spaces;

    //for (XrSpace visualizedSpace : g_visualizedSpaces) {
    //    XrSpaceLocation spaceLocation{XR_TYPE_SPACE_LOCATION};
    //    res = xrLocateSpace(visualizedSpace, g_appSpace, predictedDisplayTime, &spaceLocation);
    //    CHECK_XRRESULT(res, "xrLocateSpace");
    //    if (XR_UNQUALIFIED_SUCCESS(res)) {
    //        if ((spaceLocation.locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT) != 0 &&
    //            (spaceLocation.locationFlags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT) != 0) {
    //            spaces.push_back(Space{spaceLocation.pose, {0.25f, 0.25f, 0.25f},g_spaceNames[visualizedSpace]});;
    //        }
    //    } else {
    //        if (g_verbosity >= 2) std::cout << Fmt("Unable to locate a visualized reference space in app space: %d", res) << std::endl;
    //    }
    //}

    // Render a 10cm cube scaled by grabAction for each hand. Note renderHand will only be
    // true when the application has focus.
    /// @todo Remove these if you do not want to draw things in hand space.
    const char* handName[] = { "left", "right" };
    for (auto hand : {Side::LEFT, Side::RIGHT}) {
        XrSpaceLocation spaceLocation{XR_TYPE_SPACE_LOCATION};
        res = xrLocateSpace(g_input.handSpace[hand], g_appSpace, predictedDisplayTime, &spaceLocation);
        CHECK_XRRESULT(res, "xrLocateSpace");
        if (XR_UNQUALIFIED_SUCCESS(res)) {
            if ((spaceLocation.locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT) != 0 &&
                (spaceLocation.locationFlags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT) != 0) {
                float scale = 0.1f * g_input.handScale[hand];
                std::string name = handName[hand]; name += "Hand";
                spaces.push_back(Space{spaceLocation.pose, {scale, scale, scale},name});
            }
        } else {
            // Tracking loss is expected when the hand is not active so only log a message
            // if the hand is active.
            if (g_input.handActive[hand] == XR_TRUE) {
                if (g_verbosity >= 2) {
                    std::cout << Fmt("Unable to locate %s hand action space in app space: %d", handName[hand], res) << std::endl;
                }
            }
        }
    }

    // Render view to the appropriate part of the swapchain image.
    for (uint32_t i = 0; i < viewCountOutput; i++) {
        // Each view has a separate swapchain which is acquired, rendered to, and released.
        const Swapchain viewSwapchain = g_swapchains[i];

        XrSwapchainImageAcquireInfo acquireInfo{XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};

        uint32_t swapchainImageIndex;
        CHECK_XRCMD(xrAcquireSwapchainImage(viewSwapchain.handle, &acquireInfo, &swapchainImageIndex));

        XrSwapchainImageWaitInfo waitInfo{XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO};
        waitInfo.timeout = XR_INFINITE_DURATION;
        CHECK_XRCMD(xrWaitSwapchainImage(viewSwapchain.handle, &waitInfo));

        projectionLayerViews[i] = {XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW};
        projectionLayerViews[i].pose = g_views[i].pose;
        projectionLayerViews[i].fov = g_views[i].fov;
        projectionLayerViews[i].subImage.swapchain = viewSwapchain.handle;
        projectionLayerViews[i].subImage.imageRect.offset = {0, 0};
        projectionLayerViews[i].subImage.imageRect.extent = {viewSwapchain.width, viewSwapchain.height};

        const XrSwapchainImageBaseHeader* const swapchainImage = g_swapchainImages[viewSwapchain.handle][swapchainImageIndex];
        if(i == 0)
        {
            OpenGLRenderViewScene1(projectionLayerViews[i], swapchainImage, g_colorSwapchainFormat, spaces); // for the left eye
        }
        else
        {
            OpenGLRenderViewRight(projectionLayerViews[i], swapchainImage, g_colorSwapchainFormat, spaces); // for right eye
            //OpenGLRenderViewScene2_test(projectionLayerViews[i], swapchainImage, g_colorSwapchainFormat, spaces);
        }       

        XrSwapchainImageReleaseInfo releaseInfo{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
        CHECK_XRCMD(xrReleaseSwapchainImage(viewSwapchain.handle, &releaseInfo));
    }

    layer.space = g_appSpace;
    layer.viewCount = (uint32_t)projectionLayerViews.size();
    layer.views = projectionLayerViews.data();
    return true;
}

static void OpenXRRenderFrame()
{
    CHECK(g_session != XR_NULL_HANDLE);

    XrFrameWaitInfo frameWaitInfo{XR_TYPE_FRAME_WAIT_INFO};
    XrFrameState frameState{XR_TYPE_FRAME_STATE};
    CHECK_XRCMD(xrWaitFrame(g_session, &frameWaitInfo, &frameState));

    XrFrameBeginInfo frameBeginInfo{XR_TYPE_FRAME_BEGIN_INFO};
    CHECK_XRCMD(xrBeginFrame(g_session, &frameBeginInfo));

    std::vector<XrCompositionLayerBaseHeader*> layers;
    XrCompositionLayerProjection layer{XR_TYPE_COMPOSITION_LAYER_PROJECTION};
    std::vector<XrCompositionLayerProjectionView> projectionLayerViews;
    if (frameState.shouldRender == XR_TRUE) {
        if (OpenXRRenderLayer(frameState.predictedDisplayTime, projectionLayerViews, layer)) {
            layers.push_back(reinterpret_cast<XrCompositionLayerBaseHeader*>(&layer));
        }
    }

    XrFrameEndInfo frameEndInfo{XR_TYPE_FRAME_END_INFO};
    frameEndInfo.displayTime = frameState.predictedDisplayTime;
    frameEndInfo.environmentBlendMode = g_environmentBlendMode;
    frameEndInfo.layerCount = (uint32_t)layers.size();
    frameEndInfo.layers = layers.data();
    CHECK_XRCMD(xrEndFrame(g_session, &frameEndInfo));
}

static void OpenXRTearDown()
{
    OpenGLTearDown();

    if (g_input.actionSet != XR_NULL_HANDLE) {
        for (auto hand : {Side::LEFT, Side::RIGHT}) {
            xrDestroySpace(g_input.handSpace[hand]);
        }
        xrDestroyActionSet(g_input.actionSet);
    }

    for (Swapchain swapchain : g_swapchains) {
        xrDestroySwapchain(swapchain.handle);
    }

    for (XrSpace visualizedSpace : g_visualizedSpaces) {
        xrDestroySpace(visualizedSpace);
    }

    if (g_appSpace != XR_NULL_HANDLE) {
        xrDestroySpace(g_appSpace);
    }

    if (g_session != XR_NULL_HANDLE) {
        xrDestroySession(g_session);
    }

    if (g_instance != XR_NULL_HANDLE) {
        xrDestroyInstance(g_instance);
    }

#ifdef XR_USE_PLATFORM_WIN32
    CoUninitialize();
#endif
}

void shutdown() 
{

}
//============================================================================================
int main(int argc, char* argv[])
{
    try {
        // Parse the command line.
        int realParams = 0;
        for (int i = 1; i < argc; i++) {
            if (std::string("--verbosity") == argv[i]) {
                if (++i >= argc) { Usage(argv[0]); return 1; }
                g_verbosity = atoi(argv[i]);
            } else if (argv[i][0] == '-') {
                Usage(argv[0]); return 1;
            } else switch (++realParams) {
                case 1:
                default:
                    Usage(argv[0]);
                    return 1;
            }
        }

        // Spawn a thread to wait for a keypress
        static bool quitKeyPressed = false;
        auto exitPollingThread = std::thread{[] {
            if (g_verbosity > 0) { std::cout << "Press any key to shutdown..." << std::endl; }
            (void)getchar();
            quitKeyPressed = true;
        }};
        exitPollingThread.detach();

        bool requestRestart = false;
        do {

            // Initialize OpenXR.
            OpenXRCreateInstance();
            OpenXRInitializeSystem();
            OpenXRInitializeSession();
            OpenXRCreateSwapchains();
            
            
            printf("%s\n%s\n%s\n", glGetString(GL_VENDOR),glGetString(GL_RENDERER),glGetString(GL_VERSION));
            while (!quitKeyPressed) {
                bool exitRenderLoop = false;
                OpenXRPollEvents(&exitRenderLoop, &requestRestart);
                if (exitRenderLoop) {
                    break;
                }

                if (g_sessionRunning) {
                    OpenXRPollActions();
                    OpenXRRenderFrame();
                } else {
                    // Throttle loop since xrWaitFrame won't be called.
                    std::this_thread::sleep_for(std::chrono::milliseconds(250));
                }
            }

            OpenXRTearDown();

        } while (!quitKeyPressed && requestRestart);

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown Error" << std::endl;
        return 1;
    }
}


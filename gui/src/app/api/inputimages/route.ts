
//@ts-ignore
import { put } from '@vercel/blob';
import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const blob = await put('image.jpg', request.body, {  // 'image.jpg' is filename; use timestamp for uniqueness
      access: 'public',
      handleUploadUrl: 'https://your-app.vercel.app/api/upload',  // Optional: for resumable uploads
    });
    return NextResponse.json({ success: true, url: blob.url }, { status: 200 });
  } catch (error) {
    console.error('Upload error:', error);
    return NextResponse.json({ error: 'Upload failed' }, { status: 500 });
  }
}

export const config = {
  api: { bodyParser: false },  // Disable default body parsing for binary data
};


// import {NextRequest, NextResponse} from 'next/server'

// export async function POST(request:NextRequest) {
//   console.log("Hello!!")
//   return NextResponse.json({msg:"Hi there"})
// }